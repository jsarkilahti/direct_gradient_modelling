"""Define the shared MLP building blocks, encoder, Hamiltonian models, and neural PH components."""
from __future__ import annotations
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from oeph_definitions import MLPConfig, Conv1DConfig

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "sigmoid": nn.Sigmoid,
    "softplus": nn.Softplus,
    "identity": nn.Identity,
    "linear": nn.Identity,
}


_SOFTPLUS_BETA = 1.0


_SOFTPLUS_THRESHOLD = 20.0


def _softplus_stable(x: torch.Tensor) -> torch.Tensor:
    """Shared softplus configuration used across all classes."""
    return F.softplus(x, beta=_SOFTPLUS_BETA, threshold=_SOFTPLUS_THRESHOLD)


def _as_3d(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize input to shape [B, T, D].

    Accepted input shapes:
        - [B, D]
        - [B, T, D]
    """
    if x.dim() == 2:
        return x.unsqueeze(1)
    if x.dim() == 3:
        return x
    raise ValueError(f"Expected x with shape [B, D] or [B, T, D], got {tuple(x.shape)}")


def _restore_shape(y: torch.Tensor, original_time_dim: int) -> torch.Tensor:
    """
    Restore output shape to:
        - [B, D] if original input was [B, D]
        - [B, T, D] if original input was [B, T, D]
    """
    return y.squeeze(1) if original_time_dim == 1 else y


class _BaseGradNet(nn.Module):
    """
    Shared utility base class.

    This class is internal and only provides:
        - RNG setup
        - common shape helpers
        - common lower-bound shifting logic
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 4,
        M: int = 4,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        rng: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        floor_eps: float = 1e-6,
    ):
        """Initialize the component using the supplied configuration."""
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.M = int(M)
        self.device = device
        self.dtype = dtype
        self.floor_eps = float(floor_eps)

        if rng is None:
            rng = torch.Generator(device=device)
            if seed is not None:
                rng.manual_seed(seed)
        self.rng = rng

    @torch.no_grad()
    def enforce_lower_bound(self, x: torch.Tensor, floor: float = 0.0):
        """
        Shift the scalar potential by an additive constant so that:

            min_{samples in x} H(x) = floor

        on the provided batch.

        Important:
            - This does NOT change ∇H(x), because it only adds a constant.
            - This is batch-local, not a global lower-bound certificate.
        """
        H = self.potential(x)
        delta = floor - H.min()
        self.energy_shift.add_(delta)


class GradNetM(_BaseGradNet):
    """
    Modular GradNet-M (non-monotone, integrable).

    What this class implements
    --------------------------
    This class implements a modular GradNet-M architecture in the spirit of the
    paper's M-type construction:

        H(x) = sum_{m=1}^M rho_m(f_m(x)) + b_out^T x + 0.5 * mu * ||x||^2 + energy_shift

    with module-level scalar bottlenecks:

        f_m(x) = sum_{h=1}^H (lambda_{m,h} / B_{m,h}) * softplus(B_{m,h} * (w_{m,h}^T x + b_{m,h}))
                 + c_m

        rho_m(s) = a_m * softplus(s)

    Therefore:

        ∇H(x) = sum_m rho_m'(f_m(x)) * ∇f_m(x) + b_out + mu * x

    This is the modular architecture counterpart of the simplified 
    collapsed ridge-sum implementation.

    Constructor parameters
    ----------------------
    input_dim:
        Input dimension D.

    hidden_dim:
        Number H of ridge units inside each module.

    M:
        Number of modules.

    device, dtype:
        Tensor placement / dtype.

    rng, seed:
        Optional random generator and seed.

    floor_eps:
        Small numerical stabilizer used in divisions and positive anchors.

    Important difference from simple_GradNetM
    -----------------------------------------
    GradNetM keeps the explicit module-level scalar bottleneck:

        x -> f_m(x) -> rho_m(f_m(x))

    whereas simple_GradNetM collapses everything directly into one ridge-sum
    potential without an outer scalar transform per module.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 4,
        M: int = 4,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        rng: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        floor_eps: float = 1e-6,
    ):
        """Initialize the component using the supplied configuration."""
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            M=M,
            device=device,
            dtype=dtype,
            rng=rng,
            seed=seed,
            floor_eps=floor_eps,
        )

        # Inner ridge parameters per module.
        # Shapes:
        #   W    : [M, H, D]
        #   b    : [M, H]
        #   B    : [M, H]   (unconstrained in GradNetM)
        #   lmbd : [M, H]   (unconstrained in GradNetM)
        self.W = nn.Parameter(
            torch.randn(M, hidden_dim, input_dim, device=device, dtype=dtype, generator=self.rng)
        )
        self.b = nn.Parameter(
            torch.randn(M, hidden_dim, device=device, dtype=dtype, generator=self.rng)
        )
        self.B = nn.Parameter(
            torch.randn(M, hidden_dim, device=device, dtype=dtype, generator=self.rng)
        )
        self.lmbd = nn.Parameter(
            torch.randn(M, hidden_dim, device=device, dtype=dtype, generator=self.rng)
        )

        # Outer module transform parameters.
        #   c : [M]   module bias before outer softplus
        #   a : [M]   outer amplitude (unconstrained in GradNetM)
        self.c = nn.Parameter(
            torch.randn(M, device=device, dtype=dtype, generator=self.rng)
        )
        self.a = nn.Parameter(
            torch.randn(M, device=device, dtype=dtype, generator=self.rng)
        )

        # Global affine / quadratic anchor.
        self.b_out = nn.Parameter(torch.zeros(input_dim, device=device, dtype=dtype))
        self.mu = nn.Parameter(torch.tensor(0.0, device=device, dtype=dtype))

        # Additive constant shift of H; does not affect ∇H.
        self.energy_shift = nn.Parameter(torch.tensor(0.0, device=device, dtype=dtype))

    def _module_terms(self, x: torch.Tensor):
        """
        Compute the vectorized inner-module quantities.

        Returns
        -------
        x_3d:
            Input normalized to [B, T, D].

        x_flat:
            Flattened input [BT, D].

        z_flat:
            Ridge preactivations w^T x + b, shape [BT, MH].

        inner_potential_terms:
            Per-ridge contributions to f_m(x), shape [BT, MH].

        inner_grad_terms:
            Per-ridge coefficients multiplying W in ∇f_m(x), shape [BT, MH].

        module_scalar:
            f_m(x), shape [BT, M].

        outer_gate:
            rho_m'(f_m(x)) = a_m * sigmoid(f_m(x)), shape [BT, M].
        """
        x_3d = _as_3d(x)
        Bsz, T, D = x_3d.shape
        M, H, _ = self.W.shape

        # ---------------------------------------------------------------------
        # Flatten batch/time and flatten module/hidden dimensions.
        # Fast GEMM structure.
        # ---------------------------------------------------------------------
        x_flat = x_3d.reshape(Bsz * T, D)          # [BT, D]
        W_flat = self.W.reshape(M * H, D)          # [MH, D]
        WT = W_flat.transpose(0, 1)                # [D, MH]

        # Ridge preactivations:
        #   z_{m,h}(x) = w_{m,h}^T x + b_{m,h}
        z_flat = x_flat @ WT                       # [BT, MH]
        b_flat = self.b.reshape(M * H)             # [MH]
        z_flat = z_flat + b_flat                   # [BT, MH]

        # Unconstrained parameters in the general modular GradNet.
        B_flat = self.B.reshape(M * H)             # [MH]
        lam_flat = self.lmbd.reshape(M * H)        # [MH]
        B_eff_flat = B_flat + self.floor_eps       # [MH]

        # ---------------------------------------------------------------------
        # Inner scalar module:
        #   f_m(x) = sum_h (lambda/B_eff) * softplus(B_eff z) + c_m
        #
        # Each term is scalar-valued.
        # Summing over h produces one scalar per module.
        # ---------------------------------------------------------------------
        soft_inner = _softplus_stable(B_eff_flat * z_flat)                    # [BT, MH]
        inner_potential_terms = (lam_flat / B_eff_flat) * soft_inner

        module_scalar = inner_potential_terms.view(Bsz * T, M, H).sum(dim=2) # [BT, M]
        module_scalar = module_scalar + self.c                                # [BT, M]

        # ---------------------------------------------------------------------
        # Gradient of inner module:
        #   d/dx [ (lambda/B_eff) * softplus(B_eff z) ]
        #       = lambda * sigmoid(B_eff z) * w
        # ---------------------------------------------------------------------
        inner_grad_terms = lam_flat * torch.sigmoid(B_eff_flat * z_flat)      # [BT, MH]

        # Outer module derivative:
        #   rho_m(s) = a_m * softplus(s)
        #   rho_m'(s) = a_m * sigmoid(s)
        outer_gate = self.a.unsqueeze(0) * torch.sigmoid(module_scalar)       # [BT, M]

        return x_3d, x_flat, z_flat, inner_potential_terms, inner_grad_terms, module_scalar, outer_gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return ∇H(x).

        Input:
            x: [B, D] or [B, T, D]

        Output:
            [B, D] or [B, T, D] matching the input time layout
        """
        x_3d, _, _, _, inner_grad_terms, _, outer_gate = self._module_terms(x)
        Bsz, T, D = x_3d.shape
        M, H, _ = self.W.shape

        # Reweight inner gradients by the outer module derivative.
        outer_gate_rep = outer_gate.repeat_interleave(H, dim=1)               # [BT, MH]
        h_flat = inner_grad_terms * outer_gate_rep                            # [BT, MH]

        # Fast batched linear combination with the ridge directions w_{m,h}.
        W_flat = self.W.reshape(M * H, D)                                     # [MH, D]
        y_flat = h_flat @ W_flat                                              # [BT, D]

        # Global affine / quadratic terms.
        y = y_flat.view(Bsz, T, D)
        y = y + self.b_out + self.mu * x_3d

        return _restore_shape(y, T)

    def potential(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the scalar potential H(x).

        Potential definition
        --------------------
        H(x) = sum_m a_m * softplus(f_m(x))
             + b_out^T x
             + 0.5 * mu * ||x||^2
             + energy_shift

        where:
            f_m(x) = sum_h (lambda_{m,h}/B_{m,h}) * softplus(B_{m,h}(w_{m,h}^T x + b_{m,h})) + c_m

        Remarks
        -------
        - This is integrable by construction.
        - Since parameters are unconstrained, H need not be convex.
        - Global lower boundedness is not guaranteed unless the quadratic anchor
          dominates the negative-growth directions.
        """
        x_3d, x_flat, _, _, _, module_scalar, _ = self._module_terms(x)
        Bsz, T, _ = x_3d.shape

        phi_modules = (self.a.unsqueeze(0) * _softplus_stable(module_scalar)).sum(dim=1)  # [BT]
        phi_lin = x_flat @ self.b_out                                                      # [BT]
        phi_quad = 0.5 * self.mu * (x_flat ** 2).sum(dim=1)                                # [BT]

        phi = phi_modules + phi_lin + phi_quad + self.energy_shift
        phi = phi.view(Bsz, T)

        return phi.squeeze(1) if T == 1 else phi


class MGradNetM(_BaseGradNet):
    """
    Vectorized modular monotone GradNet-M.

    Preserves the original Hamiltonian, gradient, parameterization,
    initialization, and state_dict structure.

    Optional torch.compile acceleration:
        model.enable_compilation()
        model.disable_compilation()
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 4,
        M: int = 4,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        rng: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        floor_eps: float = 1e-6,
    ):
        """Initialize the component using the supplied configuration."""
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            M=M,
            device=device,
            dtype=dtype,
            rng=rng,
            seed=seed,
            floor_eps=floor_eps,
        )

        self.W = nn.Parameter(
            torch.randn(
                M, hidden_dim, input_dim,
                device=device, dtype=dtype, generator=self.rng
            )
        )
        self.b = nn.Parameter(
            torch.randn(
                M, hidden_dim,
                device=device, dtype=dtype, generator=self.rng
            )
        )
        self.B = nn.Parameter(
            torch.randn(
                M, hidden_dim,
                device=device, dtype=dtype, generator=self.rng
            )
        )
        self.lmbd = nn.Parameter(
            torch.randn(
                M, hidden_dim,
                device=device, dtype=dtype, generator=self.rng
            )
        )
        self.c = nn.Parameter(
            torch.randn(
                M, device=device, dtype=dtype, generator=self.rng
            )
        )
        self.a = nn.Parameter(
            torch.randn(
                M, device=device, dtype=dtype, generator=self.rng
            )
        )

        self.b_out = nn.Parameter(
            torch.zeros(input_dim, device=device, dtype=dtype)
        )
        self.mu = nn.Parameter(
            torch.tensor(0.0, device=device, dtype=dtype)
        )
        self.energy_shift = nn.Parameter(
            torch.tensor(0.0, device=device, dtype=dtype)
        )
        # ENABLE COMPILATION
        self.enable_compilation()

    def _module_terms(self, x: torch.Tensor):
        """
        Compatibility method retaining the original return structure.
        """
        x_3d = _as_3d(x)
        Bsz, T, D = x_3d.shape
        M, H, _ = self.W.shape

        x_flat = x_3d.reshape(Bsz * T, D)
        W_flat = self.W.reshape(M * H, D)

        z_flat = x_flat @ W_flat.T
        z_flat = z_flat + self.b.reshape(M * H)

        Bp_flat = (
            _softplus_stable(self.B).reshape(M * H)
            + self.floor_eps
        )
        lam_flat = _softplus_stable(self.lmbd).reshape(M * H)
        a_pos = _softplus_stable(self.a)

        u = Bp_flat * z_flat

        inner_potential_terms = (
            (lam_flat / Bp_flat) * _softplus_stable(u)
        )

        module_scalar = (
            inner_potential_terms.reshape(Bsz * T, M, H)
            .sum(dim=2)
            + self.c
        )

        inner_grad_terms = lam_flat * torch.sigmoid(u)

        outer_gate = (
            a_pos.unsqueeze(0)
            * torch.sigmoid(module_scalar)
        )

        return (
            x_3d,
            x_flat,
            z_flat,
            inner_potential_terms,
            inner_grad_terms,
            module_scalar,
            outer_gate,
            a_pos,
        )

    def enable_compilation(
        self,
        backend: str = "inductor",
        mode: str = "default",
    ) -> None:
        """
        Enable optional torch.compile acceleration.

        Call after constructing/loading the model and before training
        or benchmarking.

        Compilation overhead occurs on the first execution.
        Requires PyTorch >= 2.0.
        """
        if not hasattr(torch, "compile"):
            raise RuntimeError(
                "torch.compile requires PyTorch >= 2.0"
            )

        compiled = torch.compile(
            self._forward_impl,
            backend=backend,
            mode=mode,
        )

        object.__setattr__(
            self, "_compiled_forward", compiled
        )

    def disable_compilation(self) -> None:
        """Restore eager execution."""
        object.__setattr__(
            self, "_compiled_forward", None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the module output for its input tensor."""
        compiled = getattr(
            self, "_compiled_forward", None
        )

        if compiled is None:
            return self._forward_impl(x)

        return compiled(x)

    def _forward_impl(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluate the exact modular gradient using flattened matrix
        multiplication and broadcasting.

        Input:  [B, D] or [B, T, D]
        Output: [B, D] or [B, T, D]
        """
        x_3d = _as_3d(x)
        batch, time, dim = x_3d.shape
        modules, hidden, _ = self.W.shape

        x_flat = x_3d.reshape(batch * time, dim)
        W_flat = self.W.reshape(modules * hidden, dim)

        # All ridge preactivations in one matrix multiplication.
        z = x_flat @ W_flat.T + self.b.reshape(-1)

        # Positive parameterization.
        B_pos = (
            _softplus_stable(self.B).reshape(-1)
            + self.floor_eps
        )
        lam_pos = _softplus_stable(self.lmbd).reshape(-1)

        # Reuse the transformed preactivations.
        u = B_pos * z

        # Inner convex scalar modules.
        module_scalar = (
            (
                (lam_pos / B_pos)
                * _softplus_stable(u)
            )
            .reshape(-1, modules, hidden)
            .sum(dim=-1)
            + self.c
        )

        # Outer module derivative.
        outer = (
            _softplus_stable(self.a)
            * torch.sigmoid(module_scalar)
        ).unsqueeze(-1)

        # Inner derivatives, weighted by the outer derivatives.
        # Broadcasting replaces repeat_interleave.
        coefficients = (
            lam_pos.reshape(modules, hidden)
            * torch.sigmoid(u).reshape(
                -1, modules, hidden
            )
        ) * outer

        # Sum the ridge directions.
        y = (
            coefficients.reshape(-1, modules * hidden)
            @ W_flat
        ).reshape(batch, time, dim)

        # Affine and positive quadratic anchors.
        mu_pos = (
            _softplus_stable(self.mu)
            + self.floor_eps
        )

        y = y + self.b_out + mu_pos * x_3d

        return _restore_shape(y, time)

    def potential(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluate the original convex scalar Hamiltonian.

        H(x) = sum_m a_m^+ softplus(f_m(x))
             + b_out^T x
             + 0.5 mu_pos ||x||^2
             + energy_shift
        """
        x_3d = _as_3d(x)
        batch, time, dim = x_3d.shape
        modules, hidden, _ = self.W.shape

        x_flat = x_3d.reshape(batch * time, dim)
        W_flat = self.W.reshape(modules * hidden, dim)

        z = x_flat @ W_flat.T + self.b.reshape(-1)

        B_pos = (
            _softplus_stable(self.B).reshape(-1)
            + self.floor_eps
        )
        lam_pos = _softplus_stable(self.lmbd).reshape(-1)
        a_pos = _softplus_stable(self.a)

        u = B_pos * z

        inner = (
            (lam_pos / B_pos)
            * _softplus_stable(u)
        )

        module_scalar = (
            inner.reshape(-1, modules, hidden)
            .sum(dim=2)
            + self.c
        )

        phi_modules = (
            a_pos * _softplus_stable(module_scalar)
        ).sum(dim=1)

        phi_lin = x_flat @ self.b_out

        mu_pos = (
            _softplus_stable(self.mu)
            + self.floor_eps
        )

        phi_quad = (
            0.5 * mu_pos * (x_flat ** 2).sum(dim=1)
        )

        phi = (
            phi_modules
            + phi_lin
            + phi_quad
            + self.energy_shift
        ).reshape(batch, time)

        return phi.squeeze(1) if time == 1 else phi


class simple_MGradNetM(_BaseGradNet):
    """
    Simplified / collapsed monotone mGradNet-M.

    What this class implements
    --------------------------
    Aonotone / convex ridge-sum construction: simplified counterpart of the modular MGradNetM.

    It directly parameterizes a convex potential as a positive weighted sum of
    ridge primitives plus affine/quadratic anchor terms:

        H(x) = sum_{m,h} (a_m^+ * lambda_{m,h}^+ / B_{m,h}^+) * softplus(B_{m,h}^+(w_{m,h}^T x + b_{m,h}))
             + b_out^T x
             + 0.5 * mu_pos * ||x||^2
             + energy_shift

    Difference vs MGradNetM
    -----------------------
    MGradNetM:
        modular: one scalar inner module per module, then one outer scalar
        transform per module.

    simple_MGradNetM:
        collapsed: all ridge terms are summed directly, with no outer module
        transform.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 4,
        M: int = 4,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        rng: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        floor_eps: float = 1e-6,
    ):
        """Initialize the component using the supplied configuration."""
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            M=M,
            device=device,
            dtype=dtype,
            rng=rng,
            seed=seed,
            floor_eps=floor_eps,
        )

        self.W = nn.Parameter(
            torch.randn(M, hidden_dim, input_dim, device=device, dtype=dtype, generator=self.rng)
        )
        self.b = nn.Parameter(
            torch.randn(M, hidden_dim, device=device, dtype=dtype, generator=self.rng)
        )
        self.B = nn.Parameter(
            torch.randn(M, hidden_dim, device=device, dtype=dtype, generator=self.rng)
        )
        self.lmbd = nn.Parameter(
            torch.randn(M, hidden_dim, device=device, dtype=dtype, generator=self.rng)
        )
        self.a = nn.Parameter(
            torch.randn(M, 1, device=device, dtype=dtype, generator=self.rng)
        )

        self.b_out = nn.Parameter(torch.zeros(input_dim, device=device, dtype=dtype))
        self.mu = nn.Parameter(torch.tensor(0.0, device=device, dtype=dtype))
        self.energy_shift = nn.Parameter(torch.tensor(0.0, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return ∇H(x) for the collapsed monotone / convex potential.
        """
        x_3d = _as_3d(x)
        Bsz, T, D = x_3d.shape
        M, H, _ = self.W.shape

        x_flat = x_3d.reshape(Bsz * T, D)
        W_flat = self.W.reshape(M * H, D)
        WT = W_flat.transpose(0, 1)
        z_flat = x_flat @ WT

        b_flat = self.b.reshape(M * H)
        Bp_flat = _softplus_stable(self.B).reshape(M * H) + self.floor_eps
        lam_flat = _softplus_stable(self.lmbd).reshape(M * H)
        a_flat = _softplus_stable(self.a).squeeze(-1).repeat_interleave(H)

        z_flat = z_flat + b_flat
        h_flat = torch.sigmoid(Bp_flat * z_flat)
        h_flat = h_flat * lam_flat
        h_flat = h_flat * a_flat

        y_flat = h_flat @ W_flat
        y = y_flat.view(Bsz, T, D)

        mu_pos = _softplus_stable(self.mu) + self.floor_eps
        y = y + self.b_out + mu_pos * x_3d
        return _restore_shape(y, T)

    def potential(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the convex collapsed ridge-sum scalar potential H(x).
        """
        x_3d = _as_3d(x)
        Bsz, T, D = x_3d.shape
        M, H, _ = self.W.shape

        x_flat = x_3d.reshape(Bsz * T, D)
        W_flat = self.W.reshape(M * H, D)
        WT = W_flat.transpose(0, 1)
        z_flat = x_flat @ WT

        b_flat = self.b.reshape(M * H)
        Bp_flat = _softplus_stable(self.B).reshape(M * H) + self.floor_eps
        lam_flat = _softplus_stable(self.lmbd).reshape(M * H)
        a_flat = _softplus_stable(self.a).squeeze(-1).repeat_interleave(H)

        z_flat = z_flat + b_flat
        soft_flat = _softplus_stable(Bp_flat * z_flat)
        term_flat = (a_flat * lam_flat) / Bp_flat * soft_flat

        phi_units = term_flat.sum(dim=1)
        phi_lin = x_flat @ self.b_out
        mu_pos = _softplus_stable(self.mu) + self.floor_eps
        phi_quad = 0.5 * mu_pos * (x_flat ** 2).sum(dim=1)

        phi = phi_units + phi_lin + phi_quad + self.energy_shift
        phi = phi.view(Bsz, T)
        return phi.squeeze(1) if T == 1 else phi


def get_activation_class(name: str) -> type[nn.Module]:
    """
    Resolve an activation name to a PyTorch module class.
    """

    key = name.lower()

    if key not in _ACTIVATIONS:
        supported = ", ".join(sorted(_ACTIVATIONS))
        raise ValueError(
            f"Unknown activation '{name}'. Supported activations: {supported}."
        )

    return _ACTIVATIONS[key]


def _make_generator(
    device: str = "cpu",
    rng: Optional[torch.Generator] = None,
    seed: Optional[int] = None,
) -> torch.Generator:
    """
    Build or reuse a torch.Generator for deterministic initialization.
    """

    if rng is not None:
        return rng

    generator = torch.Generator(device=device)

    if seed is not None:
        generator.manual_seed(seed)

    return generator


def _xavier_uniform_linear_(
    layer: nn.Linear,
    rng: torch.Generator,
) -> None:
    """
    Xavier-uniform initialization for one linear layer.
    """

    fan_in = layer.weight.size(1)
    fan_out = layer.weight.size(0)
    bound = math.sqrt(6.0 / (fan_in + fan_out))

    layer.weight.uniform_(
        -bound,
        bound,
        generator=rng,
    )

    if layer.bias is not None:
        layer.bias.uniform_(
            -bound,
            bound,
            generator=rng,
        )


def _uniform_init_parameter(
    shape,
    *,
    bound: float = 1.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    rng: Optional[torch.Generator] = None,
) -> nn.Parameter:
    """
    Create a deterministically initialized parameter when a generator is supplied.
    """

    param = nn.Parameter(
        torch.empty(
            *shape,
            device=device,
            dtype=dtype,
        )
    )

    with torch.no_grad():
        param.uniform_(
            -bound,
            bound,
            generator=rng,
        )

    return param


class MLP(nn.Module):
    """
    Configurable feedforward neural network.

    The architecture is specified explicitly by ``MLPConfig.hidden_layers``.
    For example:

        MLPConfig(
            input_dim=2,
            output_dim=1,
            hidden_layers=[64, 64, 32],
            activation="tanh",
        )

    creates

        Linear(2, 64) -> Tanh
        Linear(64, 64) -> Tanh
        Linear(64, 32) -> Tanh
        Linear(32, 1)

    If ``lower_bound`` is provided, an ELU output head is appended and shifted
    by the lower-bound value.
    """

    def __init__(
        self,
        config: MLPConfig,
        *,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        rng: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
    ):
        """Initialize the component using the supplied configuration."""
        super().__init__()

        self.config = config
        self.input_dim = int(config.input_dim)
        self.output_dim = int(config.output_dim)
        self.hidden_layers = [int(width) for width in config.hidden_layers]
        self.activation = str(config.activation)
        self.rng = _make_generator(
            device=device,
            rng=rng,
            seed=seed,
        )

        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive.")

        if any(width <= 0 for width in self.hidden_layers):
            raise ValueError("All hidden layer widths must be positive.")

        if config.lower_bound is not None:
            self.register_buffer(
                "lower_bound",
                torch.tensor(
                    float(config.lower_bound),
                    dtype=dtype,
                    device=device,
                ),
            )
        else:
            self.lower_bound = None

        activation_cls = get_activation_class(self.activation)

        layer_dims = [
            self.input_dim,
            *self.hidden_layers,
            self.output_dim,
        ]

        layers: list[nn.Module] = []

        for in_dim, out_dim in zip(
            layer_dims[:-2],
            layer_dims[1:-1],
        ):
            layers.append(
                nn.Linear(
                    in_dim,
                    out_dim,
                    device=device,
                    dtype=dtype,
                )
            )
            layers.append(activation_cls())

        layers.append(
            nn.Linear(
                layer_dims[-2],
                layer_dims[-1],
                device=device,
                dtype=dtype,
            )
        )

        if config.lower_bound is not None:
            layers.append(
                nn.ELU(alpha=1.0)
            )

        self.net = nn.Sequential(*layers)
        self._init_with_rng()

    def _init_with_rng(self) -> None:
        """
        Initialize all linear layers with Xavier-uniform weights and biases.

        The final linear layer is additionally multiplied by
        ``final_layer_initialization_scale``.
        """

        scale = float(self.config.final_layer_initialization_scale)

        if scale <= 0.0:
            raise ValueError(
                "final_layer_initialization_scale must be positive."
            )

        linear_layers = [
            module
            for module in self.net
            if isinstance(module, nn.Linear)
        ]

        with torch.no_grad():
            for module in linear_layers:
                _xavier_uniform_linear_(
                    module,
                    self.rng,
                )

            final_layer = linear_layers[-1]

            final_layer.weight.mul_(scale)

            if final_layer.bias is not None:
                final_layer.bias.mul_(scale)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate the MLP.
        """

        out = self.net(x)

        if self.lower_bound is not None:
            out = out + self.lower_bound

        return out


class Conv1D(nn.Module):
    """
    Configurable one-dimensional convolutional neural network.

    Input tensors must have shape ``[batch, channels, length]``. The
    convolutional stack follows the implementation ported from ``models.py``:
    Conv1d -> activation -> optional MaxPool1d for each configured layer,
    followed by flattening and an MLP output head.
    """

    def __init__(
        self,
        config: Conv1DConfig,
        *,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        rng: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
    ):
        """Initialize the component using the supplied configuration."""
        super().__init__()

        self.config = config
        self.input_channels = int(config.input_channels)
        self.input_length = int(config.input_length)
        self.output_dim = int(config.output_dim)
        self.conv_channels = [int(value) for value in config.conv_channels]
        self.kernel_sizes = [int(value) for value in config.kernel_sizes]
        self.pool_sizes = [int(value) for value in config.pool_sizes]
        self.hidden_layers = [int(value) for value in config.hidden_layers]
        self.activation = str(config.activation)
        self.dropout = float(config.dropout)
        self.rng = _make_generator(device=device, rng=rng, seed=seed)

        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive.")
        if self.input_length <= 0:
            raise ValueError("input_length must be positive.")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        if len(self.conv_channels) == 0:
            raise ValueError("conv_channels must contain at least one layer width.")
        if not (
            len(self.conv_channels)
            == len(self.kernel_sizes)
            == len(self.pool_sizes)
        ):
            raise ValueError(
                "conv_channels, kernel_sizes, and pool_sizes must have equal length."
            )
        if any(value <= 0 for value in self.conv_channels):
            raise ValueError("All convolutional channel counts must be positive.")
        if any(value <= 0 for value in self.kernel_sizes):
            raise ValueError("All kernel sizes must be positive.")
        if any(value <= 0 for value in self.pool_sizes):
            raise ValueError("All pool sizes must be positive.")
        if any(value <= 0 for value in self.hidden_layers):
            raise ValueError("All hidden layer widths must be positive.")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")

        activation_cls = get_activation_class(self.activation)
        conv_layers: list[nn.Module] = []
        current_channels = self.input_channels
        current_length = self.input_length

        for out_channels, kernel_size, pool_size in zip(
            self.conv_channels,
            self.kernel_sizes,
            self.pool_sizes,
        ):
            padding = kernel_size // 2
            conv_layers.append(
                nn.Conv1d(
                    current_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    device=device,
                    dtype=dtype,
                )
            )
            conv_layers.append(activation_cls())

            if pool_size > 1:
                conv_layers.append(
                    nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
                )
                current_length = current_length // pool_size
                if current_length <= 0:
                    raise ValueError(
                        "Configured pooling reduces the convolutional input length to zero."
                    )

            current_channels = out_channels

        self.conv = nn.Sequential(*conv_layers)
        self.flattened_dim = int(current_channels * current_length)

        head_dims = [self.flattened_dim, *self.hidden_layers, self.output_dim]
        head_layers: list[nn.Module] = []
        for index, (in_dim, out_dim) in enumerate(
            zip(head_dims[:-1], head_dims[1:])
        ):
            head_layers.append(
                nn.Linear(in_dim, out_dim, device=device, dtype=dtype)
            )
            if index < len(head_dims) - 2:
                head_layers.append(activation_cls())
                if self.dropout > 0.0:
                    head_layers.append(nn.Dropout(p=self.dropout))

        self.head = nn.Sequential(*head_layers)
        self._init_with_rng()

    def _init_with_rng(self) -> None:
        """Handle init with rng for the experiment workflow. Used in the neural PH model architecture."""
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Conv1d):
                    nn.init.kaiming_uniform_(
                        module.weight,
                        nonlinearity="relu",
                        generator=self.rng,
                    )
                    if module.bias is not None:
                        module.bias.zero_()
                elif isinstance(module, nn.Linear):
                    _xavier_uniform_linear_(module, self.rng)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the module output for its input tensor."""
        if x.dim() != 3:
            raise ValueError(
                "Expected input shape [batch, channels, length], "
                f"got {tuple(x.shape)}."
            )
        if tuple(x.shape[1:]) != (self.input_channels, self.input_length):
            raise ValueError(
                "Unexpected Conv1D input shape "
                f"{tuple(x.shape)}; expected [batch, {self.input_channels}, "
                f"{self.input_length}]."
            )

        features = self.conv(x)
        features = features.reshape(features.shape[0], -1)
        return self.head(features)


class LegacyHNet(nn.Module):
    """
    Legacy-style scalar Hamiltonian network.

    This class is retained as a compatibility building block. It now consumes an
    ``MLPConfig`` rather than separate ``num_hidden`` and ``num_neurons``
    arguments.
    """

    def __init__(
        self,
        config: MLPConfig,
        *,
        lower_bound: float = 0.0,
        use_quadratic_term: bool = False,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        rng: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
    ):
        """Initialize the component using the supplied configuration."""
        super().__init__()

        if config.output_dim != 1:
            raise ValueError(
                "LegacyHNet requires config.output_dim == 1."
            )

        self.config = config
        self.use_quadratic_term = bool(use_quadratic_term)
        self.rng = _make_generator(
            device=device,
            rng=rng,
            seed=seed,
        )

        h_config = MLPConfig(
            input_dim=config.input_dim,
            output_dim=1,
            hidden_layers=list(config.hidden_layers),
            activation=config.activation,
            lower_bound=None,
        )

        self.h_non = MLP(
            h_config,
            device=device,
            dtype=dtype,
            rng=self.rng,
        )

        if self.use_quadratic_term:
            self.Q_raw = nn.Parameter(
                torch.empty(
                    config.input_dim,
                    config.input_dim,
                    device=device,
                    dtype=dtype,
                )
            )
            with torch.no_grad():
                bound = math.sqrt(
                    6.0 / (2 * config.input_dim)
                )
                self.Q_raw.uniform_(
                    -bound,
                    bound,
                    generator=self.rng,
                )

        self.register_buffer(
            "lower_bound",
            torch.tensor(
                float(lower_bound),
                dtype=dtype,
                device=device,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate the legacy Hamiltonian network.
        """

        h = self.h_non(x)

        if self.use_quadratic_term:
            Q = torch.matmul(
                self.Q_raw,
                self.Q_raw.transpose(0, 1),
            )
            quad = torch.sum(
                (x @ Q) * x,
                dim=1,
                keepdim=True,
            )
            h = h + quad

        b = self.lower_bound + 1.0
        return torch.nn.functional.elu(h - b) + b


# PH components (exact numerical implementations from ph_components.py)
from oeph_definitions import (GradNetHamiltonianConfig, LegacyHamiltonianConfig, NeuralNetworkDefinition, HistoryEncoderConfig, NeuralSkewMatrixConfig, ConstantSkewMatrixConfig, NeuralPositiveSemidefiniteMatrixConfig, ConstantPositiveSemidefiniteMatrixConfig, NeuralPortMatrixConfig, ConstantPortMatrixConfig)

class GradNetHamiltonian(nn.Module):
    """Scientific wrapper around a GradNet Hamiltonian."""

    def __init__(
        self,
        config: GradNetHamiltonianConfig,
        *,
        device="cpu",
        dtype=torch.float32,
        rng=None,
    ):
        """Initialize the component using the supplied configuration."""
        super().__init__()

        self.network = config.gradnet_type(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            M=config.num_modules,
            floor_eps=config.floor_eps,
            device=device,
            dtype=dtype,
            rng=rng,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the module output for its input tensor."""
        return self.network(x)

    def potential(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the neural Hamiltonian potential."""
        potential = getattr(self.network, "potential", None)
        if potential is None:
            raise RuntimeError(
                "The wrapped GradNet does not expose potential(x)."
            )
        return potential(x)


class LegacyHamiltonian(nn.Module):

    def __init__(
        self,
        config,
        *,
        device="cpu",
        dtype=torch.float32,
        rng=None,
    ):
        """Initialize the component using the supplied configuration."""
        super().__init__()

        self.network = LegacyHNet(
            config.network.network_architecture,
            lower_bound=config.lower_bound,
            device=device,
            dtype=dtype,
            rng=rng,
        )

    def forward(self, x):
        """Compute the module output for its input tensor."""
        return self.network(x)


class HistoryEncoder(nn.Module):
    """Encode concatenated control/output histories into one latent state."""

    def __init__(
        self,
        config: HistoryEncoderConfig,
        *,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        rng: torch.Generator | None = None,
    ) -> None:
        """Initialize the component using the supplied configuration."""
        super().__init__()
        self.config = config
        self.history_controls = int(config.history_controls)
        self.history_outputs = int(config.history_outputs)
        self.input_dim = int(config.input_dim)
        self.output_dim = int(config.output_dim)
        self.state_dim = int(config.state_dim)
        self.network = _build_network(config.network, device=device, dtype=dtype, rng=rng)

    def forward(
        self,
        history_controls: torch.Tensor,
        history_outputs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the module output for its input tensor."""
        if history_controls.dim() != 3:
            raise ValueError("history_controls must have shape [batch, history, input_dim].")
        if history_outputs.dim() != 3:
            raise ValueError("history_outputs must have shape [batch, history, output_dim].")
        if history_controls.shape[1:] != (self.history_controls, self.input_dim):
            raise ValueError(
                "Unexpected history_controls shape "
                f"{tuple(history_controls.shape)}."
            )
        if history_outputs.shape[1:] != (self.history_outputs, self.output_dim):
            raise ValueError(
                "Unexpected history_outputs shape "
                f"{tuple(history_outputs.shape)}."
            )
        if history_controls.shape[0] != history_outputs.shape[0]:
            raise ValueError("Input and output histories must have equal batch size.")

        batch_size = history_controls.shape[0]

        if isinstance(self.network, Conv1D):
            network_input = torch.cat(
                [
                    history_controls.transpose(1, 2),
                    history_outputs.transpose(1, 2),
                ],
                dim=1,
            )
        else:
            network_input = torch.cat(
                [
                    history_controls.reshape(batch_size, -1),
                    history_outputs.reshape(batch_size, -1),
                ],
                dim=1,
            )

        state = self.network(network_input)
        if tuple(state.shape) != (batch_size, self.state_dim):
            raise ValueError(
                f"Encoder returned shape {tuple(state.shape)}; expected "
                f"({batch_size}, {self.state_dim})."
            )
        return state


class NeuralSkewMatrix(nn.Module):
    """Return J(x) = B(x) - B(x)^T."""

    def __init__(self, config: NeuralSkewMatrixConfig, *, device="cpu", dtype=torch.float32, rng=None):
        """Initialize the component using the supplied configuration."""
        super().__init__()
        self.state_dim = int(config.state_dim)
        self.network = _build_network(config.network, device=device, dtype=dtype, rng=rng)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the module output for its input tensor."""
        _validate_state(x, self.state_dim)
        raw = self.network(x)
        _require_flat_output(raw, x.shape[0], self.state_dim**2, "NeuralSkewMatrix")
        matrix = raw.reshape(x.shape[0], self.state_dim, self.state_dim)
        return matrix - matrix.transpose(1, 2)


class ConstantSkewMatrix(nn.Module):
    """Return one constant matrix J = B - B^T for every state."""

    def __init__(self, config: ConstantSkewMatrixConfig, *, device="cpu", dtype=torch.float32, rng=None):
        """Initialize the component using the supplied configuration."""
        super().__init__()
        self.state_dim = int(config.state_dim)
        raw = _initial_tensor(
            (self.state_dim, self.state_dim),
            config.initial_matrix,
            config.initialization_scale,
            device,
            dtype,
            rng,
        )
        _register_tensor(self, "raw_matrix", raw, config.trainable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the module output for its input tensor."""
        _validate_state(x, self.state_dim)
        matrix = self.raw_matrix - self.raw_matrix.transpose(0, 1)
        return matrix.unsqueeze(0).expand(x.shape[0], -1, -1)


class NeuralPositiveSemidefiniteMatrix(nn.Module):
    """Return R(x) = A(x) A(x)^T."""

    def __init__(self, config: NeuralPositiveSemidefiniteMatrixConfig, *, device="cpu", dtype=torch.float32, rng=None):
        """Initialize the component using the supplied configuration."""
        super().__init__()
        self.state_dim = int(config.state_dim)
        self.factor_rank = self.state_dim if config.factor_rank is None else int(config.factor_rank)
        self.network = _build_network(config.network, device=device, dtype=dtype, rng=rng)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the module output for its input tensor."""
        _validate_state(x, self.state_dim)
        raw = self.network(x)
        _require_flat_output(
            raw,
            x.shape[0],
            self.state_dim * self.factor_rank,
            "NeuralPositiveSemidefiniteMatrix",
        )
        factor = raw.reshape(x.shape[0], self.state_dim, self.factor_rank)
        return torch.bmm(factor, factor.transpose(1, 2))


class ConstantPositiveSemidefiniteMatrix(nn.Module):
    """Return one constant matrix R = A A^T for every state."""

    def __init__(self, config: ConstantPositiveSemidefiniteMatrixConfig, *, device="cpu", dtype=torch.float32, rng=None):
        """Initialize the component using the supplied configuration."""
        super().__init__()
        self.state_dim = int(config.state_dim)
        self.factor_rank = self.state_dim if config.factor_rank is None else int(config.factor_rank)
        factor = _initial_tensor(
            (self.state_dim, self.factor_rank),
            config.initial_factor,
            config.initialization_scale,
            device,
            dtype,
            rng,
        )
        _register_tensor(self, "factor", factor, config.trainable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the module output for its input tensor."""
        _validate_state(x, self.state_dim)
        matrix = self.factor @ self.factor.transpose(0, 1)
        return matrix.unsqueeze(0).expand(x.shape[0], -1, -1)


class NeuralPortMatrix(nn.Module):
    """Return a state-dependent port matrix G(x)."""

    def __init__(self, config: NeuralPortMatrixConfig, *, device="cpu", dtype=torch.float32, rng=None):
        """Initialize the component using the supplied configuration."""
        super().__init__()
        self.state_dim = int(config.state_dim)
        self.input_dim = int(config.input_dim)
        self.network = _build_network(config.network, device=device, dtype=dtype, rng=rng)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the module output for its input tensor."""
        _validate_state(x, self.state_dim)
        raw = self.network(x)
        _require_flat_output(
            raw,
            x.shape[0],
            self.state_dim * self.input_dim,
            "NeuralPortMatrix",
        )
        return raw.reshape(x.shape[0], self.state_dim, self.input_dim)


class ConstantPortMatrix(nn.Module):
    """Return one constant port matrix for every state."""

    def __init__(self, config: ConstantPortMatrixConfig, *, device="cpu", dtype=torch.float32, rng=None):
        """Initialize the component using the supplied configuration."""
        super().__init__()
        self.state_dim = int(config.state_dim)
        self.input_dim = int(config.input_dim)
        matrix = _initial_tensor(
            (self.state_dim, self.input_dim),
            config.initial_matrix,
            config.initialization_scale,
            device,
            dtype,
            rng,
        )
        _register_tensor(self, "matrix", matrix, config.trainable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the module output for its input tensor."""
        _validate_state(x, self.state_dim)
        return self.matrix.unsqueeze(0).expand(x.shape[0], -1, -1)


def _build_network(definition, *, device, dtype, rng):
    """Construct network from the supplied configuration. Used in the neural PH model architecture."""
    return definition.network_type(
        definition.network_architecture,
        device=device,
        dtype=dtype,
        rng=rng,
    )


def _require_architecture_dimension(architecture, *, name, expected, component):
    """Validate architecture dimension against the required model dimensions. Used in the neural PH model architecture."""
    if not hasattr(architecture, name):
        raise TypeError(f"{component} requires architecture.{name}.")
    actual = int(getattr(architecture, name))
    if actual != int(expected):
        raise ValueError(
            f"{component} requires architecture.{name} == {expected}, got {actual}."
        )


def _validate_state(x: torch.Tensor, state_dim: int) -> None:
    """Validate state against the required model dimensions. Used in the neural PH model architecture."""
    if x.dim() != 2 or x.shape[1] != state_dim:
        raise ValueError(
            f"Expected state shape [batch, {state_dim}], got {tuple(x.shape)}."
        )


def _require_flat_output(output, batch_size, output_dim, component):
    """Validate flat output against the required model dimensions. Used in the neural PH model architecture."""
    if tuple(output.shape) != (batch_size, output_dim):
        raise ValueError(
            f"{component} returned shape {tuple(output.shape)}; expected "
            f"({batch_size}, {output_dim})."
        )


def _validate_optional_shape(value, expected_shape, name):
    """Validate optional shape against the required model dimensions. Used in the neural PH model architecture."""
    if value is None:
        return
    if tuple(torch.as_tensor(value).shape) != tuple(expected_shape):
        raise ValueError(f"{name} must have shape {expected_shape}.")


def _initial_tensor(shape, supplied, scale, device, dtype, rng):
    """Handle initial tensor for the experiment workflow. Used in the neural PH model architecture."""
    if supplied is not None:
        return torch.as_tensor(supplied, device=device, dtype=dtype).clone()
    generator = rng if rng is not None else torch.Generator(device=device)
    return float(scale) * torch.randn(
        *shape,
        device=device,
        dtype=dtype,
        generator=generator,
    )


def _register_tensor(module, name, tensor, trainable):
    """Handle register tensor for the experiment workflow. Used in the neural PH model architecture."""
    if trainable:
        module.register_parameter(name, nn.Parameter(tensor))
    else:
        module.register_buffer(name, tensor)

