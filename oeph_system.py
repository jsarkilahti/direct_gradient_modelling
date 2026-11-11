"""Assemble PH models and shared neural components from experiment configurations."""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable
import torch
import torch.nn as nn
from oeph_definitions import ExperimentConfig
from oeph_neural_models import (MLP, Conv1D, LegacyHamiltonian, GradNetHamiltonian, HistoryEncoder, NeuralSkewMatrix, NeuralPositiveSemidefiniteMatrix, NeuralPortMatrix, ConstantSkewMatrix, ConstantPositiveSemidefiniteMatrix, ConstantPortMatrix, GradNetM, MGradNetM, simple_MGradNetM)
from oeph_definitions import (MLPConfig, Conv1DConfig, NeuralNetworkDefinition, HistoryEncoderConfig, NeuralSkewMatrixConfig, NeuralPositiveSemidefiniteMatrixConfig, NeuralPortMatrixConfig, ConstantSkewMatrixConfig, ConstantPositiveSemidefiniteMatrixConfig, ConstantPortMatrixConfig, LegacyHamiltonianConfig, GradNetHamiltonianConfig)

Dynamics = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class BaseIntegrator(nn.Module, ABC):
    """Abstract base class for one-step ODE integrators."""

    @abstractmethod
    def step(
        self,
        dynamics: Dynamics,
        x: torch.Tensor,
        u: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """Advance ``x`` one timestep under zero-order-hold input ``u``."""
        raise NotImplementedError


class EulerIntegrator(BaseIntegrator):
    """First-order explicit Euler integrator."""

    def step(self, dynamics: Dynamics, x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
        """Advance the state by one integration step."""
        return x + dt * dynamics(x, u)


class RK4Integrator(BaseIntegrator):
    """Classical fourth-order Runge-Kutta integrator."""

    def step(self, dynamics: Dynamics, x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
        """Advance the state by one integration step."""
        k1 = dynamics(x, u)
        k2 = dynamics(x + 0.5 * dt * k1, u)
        k3 = dynamics(x + 0.5 * dt * k2, u)
        k4 = dynamics(x + dt * k3, u)

        return x + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


class RK8Integrator(BaseIntegrator):
    """
    Eighth-order explicit Runge-Kutta integrator.

    This uses the 12-stage DOP853/Dormand-Prince 8 tableau for the eighth-order
    solution. Error estimation and adaptive step-size control are intentionally
    not included here; PHModel currently needs a fixed-step one-step map.
    """

    # Coefficients match the DOP853 eighth-order tableau used by SciPy.
    _A = (
        (),
        (5.260015195876773e-2,),
        (1.9725056984537898e-2, 5.917517095361369e-2),
        (2.9587585476806844e-2, 0.0, 8.876275643042054e-2),
        (2.413651341592667e-1, 0.0, -8.84549479328286e-1, 9.24834003261792e-1),
        (3.7037037037037035e-2, 0.0, 0.0, 1.7082860872947386e-1, 1.2546768756682242e-1),
        (3.7109375e-2, 0.0, 0.0, 1.7025221101954404e-1, 6.021653898045596e-2, -1.7578125e-2),
        (3.709200011850479e-2, 0.0, 0.0, 1.7038392571223998e-1, 1.0726203044637328e-1, -1.5319437748624402e-2, 8.273789163814023e-3),
        (6.241109587160757e-1, 0.0, 0.0, -3.3608926294469414, -8.682193468417260e-1, 2.7592099699446707e1, 2.0154067550477893e1, -4.348988418106995e1),
        (4.776625364382643e-1, 0.0, 0.0, -2.4881146199716677, -5.902908268368429e-1, 2.1230051448181195e1, 1.5279233632882424e1, -3.328821096898486e1, -2.0331201708508626e-2),
        (-9.371424300859873e-1, 0.0, 0.0, 5.186372428844064, 1.0914373489967295, -8.149787010746927, -1.8520065659996959e1, 2.2739487099350505e1, 2.4936055526796523, -3.0467644718982195),
        (2.273310147516538, 0.0, 0.0, -1.053449546673725e1, -2.0008720582248625, -1.7958931863118799e1, 2.794888452941996e1, -2.8589982771350235, -8.87285693353063, 1.2360567175794303e1, 6.433927460157635e-1),
    )
    _B = (
        5.429373411656876e-2,
        0.0,
        0.0,
        0.0,
        0.0,
        4.450312892752409,
        1.8915178993145003,
        -5.801203960010585,
        3.111643669578199e-1,
        -1.521609496625161e-1,
        2.0136540080403034e-1,
        4.471061572777259e-2,
    )

    def step(self, dynamics: Dynamics, x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
        """Advance the state by one integration step."""
        stages: list[torch.Tensor] = []

        for row in self._A:
            if len(row) == 0:
                x_stage = x
            else:
                increment = sum(
                    coeff * stage
                    for coeff, stage in zip(row, stages)
                    if coeff != 0.0
                )
                x_stage = x + dt * increment

            stages.append(dynamics(x_stage, u))

        weighted = sum(
            coeff * stage
            for coeff, stage in zip(self._B, stages)
            if coeff != 0.0
        )

        return x + dt * weighted


def make_integrator(name: str) -> BaseIntegrator:
    """Instantiate an integrator by name."""

    key = str(name).lower()

    if key == "euler":
        return EulerIntegrator()

    if key == "rk4":
        return RK4Integrator()

    if key == "rk8":
        return RK8Integrator()

    raise ValueError(f"Unknown integrator '{name}'.")


class PHModel(nn.Module):
    """Assemble and evaluate one already-constructed PH neural model.

    Parameters
    ----------
    encoder
        Module called as ``encoder(history_controls, history_outputs)`` and
        returning the initial latent state ``[batch, state_dim]``.

    interconnection
        Module implementing ``J(x)`` and returning
        ``[batch, state_dim, state_dim]``.

    dissipation
        Module implementing ``R(x)`` and returning
        ``[batch, state_dim, state_dim]``.

    port
        Module implementing ``G(x)`` and returning
        ``[batch, state_dim, input_dim]``.

    hamiltonian
        For ``hamiltonian_representation="autograd"``, a scalar Hamiltonian
        module ``H(x)``.  For ``"gradnet"``, a module returning ``grad H(x)``.
        A GradNet may additionally expose ``potential(x)``.

    integrator
        Already-constructed one-step numerical integrator.

    input_dim, output_dim, state_dim
        Scientific dimensions of the model.

    sampling_time
        Sampling interval used by one model step.

    history_controls, history_outputs
        Required input and output history lengths of the encoder.

    hamiltonian_representation
        Either ``"autograd"`` or ``"gradnet"``.

    integrator_substeps
        Fixed integration substeps per sampling interval.

    gradient_scale
        Optional multiplicative scale applied to a directly represented
        Hamiltonian gradient.
    """

    def __init__(
        self,
        *,
        encoder: nn.Module,
        interconnection: nn.Module,
        dissipation: nn.Module,
        port: nn.Module,
        hamiltonian: nn.Module,
        integrator: BaseIntegrator,
        input_dim: int,
        output_dim: int,
        state_dim: int,
        sampling_time: float,
        history_controls: int,
        history_outputs: int,
        hamiltonian_representation: str,
        integrator_substeps: int = 1,
        gradient_scale: float = 1.0,
        initial_state_source: str = "state_encoder",
        output_before_integration: bool = False,
    ) -> None:
        """Initialize the component using the supplied configuration."""
        super().__init__()

        self.output_before_integration = bool(output_before_integration)
        self.encoder = encoder
        self.interconnection = interconnection
        self.dissipation = dissipation
        self.port = port
        self.hamiltonian_component = hamiltonian
        self.integrator = integrator

        self.dim_u = int(input_dim)
        self.dim_y = int(output_dim)
        self.dim_x = int(state_dim)

        if self.dim_u <= 0 or self.dim_y <= 0 or self.dim_x <= 0:
            raise ValueError("input_dim, output_dim, and state_dim must be positive.")

        if self.dim_y != self.dim_u:
            raise ValueError(
                "The output map y = G(x)^T grad H(x) requires "
                "output_dim == input_dim."
            )

        self.Ts = float(sampling_time)
        if self.Ts <= 0.0:
            raise ValueError("sampling_time must be positive.")

        self.n_u = int(history_controls)
        self.n_y = int(history_outputs)
        if self.n_u <= 0 or self.n_y <= 0:
            raise ValueError("history_controls and history_outputs must be positive.")

        self.integrator_substeps = int(integrator_substeps)
        if self.integrator_substeps <= 0:
            raise ValueError("integrator_substeps must be positive.")

        self.hamiltonian_representation = str(
            hamiltonian_representation
        ).lower()
        if self.hamiltonian_representation not in {"autograd", "gradnet"}:
            raise ValueError(
                "hamiltonian_representation must be 'autograd' or 'gradnet'."
            )

        self.gradient_scale = float(gradient_scale)

        self.initial_state_source = str(initial_state_source).lower()
        if self.initial_state_source not in {"state_encoder", "true_ph_state"}:
            raise ValueError(
                "initial_state_source must be 'state_encoder' or 'true_ph_state'."
            )

        self.state_derivative_limit = float("inf")

    @property
    def requires_forward_grad(self) -> bool:
        """Whether rollout evaluation must keep gradients enabled."""

        return self.hamiltonian_representation == "autograd"

    def encode(
        self,
        history_controls: torch.Tensor,
        history_outputs: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate the initial latent state from the supplied history."""

        if history_controls.dim() != 3:
            raise ValueError(
                "history_controls must have shape [batch, history, input_dim]."
            )
        if history_outputs.dim() != 3:
            raise ValueError(
                "history_outputs must have shape [batch, history, output_dim]."
            )
        if history_controls.shape[1] != self.n_u:
            raise ValueError(
                f"Expected history_controls.shape[1] == {self.n_u}, "
                f"got {history_controls.shape[1]}."
            )
        if history_outputs.shape[1] != self.n_y:
            raise ValueError(
                f"Expected history_outputs.shape[1] == {self.n_y}, "
                f"got {history_outputs.shape[1]}."
            )
        if history_controls.shape[2] != self.dim_u:
            raise ValueError(
                f"Expected history_controls.shape[2] == {self.dim_u}, "
                f"got {history_controls.shape[2]}."
            )
        if history_outputs.shape[2] != self.dim_y:
            raise ValueError(
                f"Expected history_outputs.shape[2] == {self.dim_y}, "
                f"got {history_outputs.shape[2]}."
            )
        if history_controls.shape[0] != history_outputs.shape[0]:
            raise ValueError("Input and output histories must have equal batch size.")

        state = self.encoder(history_controls, history_outputs)
        self._require_shape(state, (history_controls.shape[0], self.dim_x), "encoder")
        return state

    def hamiltonian(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the scalar Hamiltonian when the component exposes it."""

        self._validate_state(x)

        if self.hamiltonian_representation == "autograd":
            return self.hamiltonian_component(x)

        potential = getattr(self.hamiltonian_component, "potential", None)
        if potential is None:
            raise RuntimeError(
                "The supplied gradient-network Hamiltonian does not expose "
                "potential(x)."
            )

        try:
            return potential(x)
        except ValueError:
            if x.dim() == 2:
                return potential(x.unsqueeze(1))
            raise

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the Hamiltonian gradient ``grad H(x)``."""

        self._validate_state(x)

        if self.hamiltonian_representation == "autograd":
            if not x.requires_grad:
                x = x.requires_grad_(True)

            energy = self.hamiltonian_component(x)
            gradient = torch.autograd.grad(
                energy.sum(),
                x,
                create_graph=True,
            )[0]
        else:
            gradient = self.hamiltonian_component(x)
            if self.gradient_scale != 1.0:
                gradient = self.gradient_scale * gradient

        self._require_shape(gradient, x.shape, "Hamiltonian gradient")
        return gradient

    def J(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the supplied skew-symmetric interconnection matrix."""

        self._validate_state(x)
        matrix = self.interconnection(x)
        self._require_matrix_shape(matrix, x.shape[0], "J")
        return matrix

    def R(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the supplied positive-semidefinite dissipation matrix."""

        self._validate_state(x)
        matrix = self.dissipation(x)
        self._require_matrix_shape(matrix, x.shape[0], "R")
        return matrix

    def G(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the supplied input/output port matrix."""

        self._validate_state(x)
        matrix = self.port(x)
        self._require_shape(
            matrix,
            (x.shape[0], self.dim_x, self.dim_u),
            "G",
        )
        return matrix

    def dynamics(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the continuous-time port-Hamiltonian dynamics."""

        self._validate_state(x)
        self._validate_input(u, x.shape[0])

        G = self.G(x)
        grad_H = self.gradient(x)

        autonomous = torch.bmm(
            self.J(x) - self.R(x),
            grad_H.unsqueeze(-1),
        ).squeeze(-1)

        controlled = torch.bmm(
            G,
            u.unsqueeze(-1),
        ).squeeze(-1)

        dx = autonomous + controlled

        if self.state_derivative_limit < float("inf"):
            finfo = torch.finfo(dx.dtype)

            # Convert overflowed components to finite signed values before
            # normalization. NaNs carry no usable direction information.
            dx_safe = torch.nan_to_num(
                dx,
                nan=0.0,
                posinf=finfo.max,
                neginf=-finfo.max,
            )

            # Normalize through max-absolute scaling so computing the Euclidean
            # norm itself cannot overflow.
            max_abs = dx_safe.abs().amax(dim=-1, keepdim=True)

            normalized = dx_safe / torch.clamp(max_abs, min=finfo.tiny)

            normalized_norm = torch.linalg.vector_norm(
                normalized,
                ord=2,
                dim=-1,
                keepdim=True,
            )

            unit_direction = normalized / torch.clamp(
                normalized_norm,
                min=finfo.tiny,
            )

            original_norm_safe = max_abs * normalized_norm

            limited_norm = torch.clamp(
                original_norm_safe,
                max=self.state_derivative_limit,
            )

            dx = unit_direction * limited_norm

        if self.state_derivative_limit < float("inf"):
            if not torch.all(torch.isfinite(dx)):
                raise FloatingPointError(
                    f"DERIVATIVE LIMITER RETURNED NON-FINITE dx\n"
                    f"limit={self.state_derivative_limit}\n"
                    f"dx={dx}"
                )

            dx_norm = torch.linalg.vector_norm(dx, ord=2, dim=-1)

            if torch.any(dx_norm > self.state_derivative_limit * 1.0001):
                raise RuntimeError(
                    f"DERIVATIVE LIMITER FAILED TO ENFORCE LIMIT\n"
                    f"limit={self.state_derivative_limit}\n"
                    f"max_norm={dx_norm.max().item()}"
                )

        return dx


    def output(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate ``y = G(x)^T grad H(x)``."""

        self._validate_state(x)
        output = torch.bmm(
            self.G(x).transpose(1, 2),
            self.gradient(x).unsqueeze(-1),
        ).squeeze(-1)
        self._require_shape(output, (x.shape[0], self.dim_y), "output")
        return output

    def step(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        """Advance one sampling interval under zero-order-hold input."""

        self._validate_state(x)
        self._validate_input(u, x.shape[0])

        dt = self.Ts / self.integrator_substeps

        for _ in range(self.integrator_substeps):
            x = self.integrator.step(
                self.dynamics,
                x,
                u,
                dt,
            )

        return x

    def rollout(
        self,
        history_controls: torch.Tensor,
        history_outputs: torch.Tensor,
        future_controls: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Roll out causally aligned future output predictions."""

        if future_controls.dim() != 3:
            raise ValueError(
                "future_controls must have shape [batch, horizon, input_dim]."
            )
        if future_controls.shape[2] != self.dim_u:
            raise ValueError(
                f"Expected future_controls.shape[2] == {self.dim_u}, "
                f"got {future_controls.shape[2]}."
            )
        if future_controls.shape[0] != history_controls.shape[0]:
            raise ValueError(
                "future_controls and history_controls must have equal batch size."
            )
        if future_controls.shape[1] <= 0:
            raise ValueError("future_controls must contain at least one step.")

        if self.initial_state_source == "state_encoder":
            if initial_state is not None:
                raise ValueError(
                    "initial_state must be None when initial_state_source='state_encoder'."
                )
            x = self.encode(history_controls, history_outputs)
        else:
            if initial_state is None:
                raise ValueError(
                    "initial_state is required when initial_state_source='true_ph_state'."
                )
            x = initial_state.to(
                device=future_controls.device,
                dtype=future_controls.dtype,
            )
            self._require_shape(
                x,
                (future_controls.shape[0], self.dim_x),
                "true PH initial state",
            )

        predictions: list[torch.Tensor] = []

        for time_index in range(future_controls.shape[1]):
            if self.output_before_integration:
                # Encoder estimates the first target state from strictly past data.
                predictions.append(self.output(x).unsqueeze(1))
                if time_index + 1 < future_controls.shape[1]:
                    x = self.step(x, future_controls[:, time_index, :])
            else:
                x = self.step(x, future_controls[:, time_index, :])
                predictions.append(self.output(x).unsqueeze(1))

        return torch.cat(predictions, dim=1)

    def forward(
        self,
        history_controls: torch.Tensor,
        history_outputs: torch.Tensor,
        future_controls: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Alias for :meth:`rollout`."""

        return self.rollout(
            history_controls,
            history_outputs,
            future_controls,
            initial_state=initial_state,
        )

    def _validate_state(self, x: torch.Tensor) -> None:
        """Validate state against the required model dimensions. Used to construct or execute a PH model."""
        if x.dim() != 2 or x.shape[1] != self.dim_x:
            raise ValueError(
                f"Expected state shape [batch, {self.dim_x}], got {tuple(x.shape)}."
            )

    def _validate_input(self, u: torch.Tensor, batch_size: int) -> None:
        """Validate input against the required model dimensions. Used to construct or execute a PH model."""
        if u.dim() != 2 or u.shape != (batch_size, self.dim_u):
            raise ValueError(
                f"Expected input shape [{batch_size}, {self.dim_u}], "
                f"got {tuple(u.shape)}."
            )

    def _require_matrix_shape(
        self,
        tensor: torch.Tensor,
        batch_size: int,
        name: str,
    ) -> None:
        """Validate matrix shape against the required model dimensions. Used to construct or execute a PH model."""
        self._require_shape(
            tensor,
            (batch_size, self.dim_x, self.dim_x),
            name,
        )

    @staticmethod
    def _require_shape(
        tensor: torch.Tensor,
        expected_shape: tuple[int, ...] | torch.Size,
        name: str,
    ) -> None:
        """Validate shape against the required model dimensions. Used to construct or execute a PH model."""
        if tuple(tensor.shape) != tuple(expected_shape):
            raise ValueError(
                f"{name} returned shape {tuple(tensor.shape)}; "
                f"expected {tuple(expected_shape)}."
            )



def build_model(config: ExperimentConfig, method: str, model_seed: int,
                component_seeds: dict[str, int] | None = None) -> PHModel:
    """Construct one OE-pHNN using the reference component initialization order.

    component_seeds may override the common seed for independent-component studies.
    Seed derivation for study sweeps belongs to the later study runner.
    """
    m, w, d = config.model, config.windows, config.data
    if method not in ('autograd', 'mgradnet_m', 'simple_mgradnet_m', 'gradnet_m'):
        raise ValueError(f'Unknown method: {method}')
    dtype = getattr(torch, m.dtype)
    device = m.device
    seeds = dict(component_seeds or {})
    def generator(name):
        """Handle generator for the experiment workflow. Used to construct or execute a PH model."""
        g = torch.Generator(device=device)
        g.manual_seed(int(seeds.get(name, model_seed)))
        return g
    def mlp(n_in, n_out, hidden):
        """Handle mlp for the experiment workflow. Used to construct or execute a PH model."""
        return NeuralNetworkDefinition(MLP, MLPConfig(
            input_dim=n_in, output_dim=n_out, hidden_layers=list(hidden),
            activation=m.activation,
            final_layer_initialization_scale=m.final_layer_initialization_scale))
    nx, nu, ny = m.state_dim, m.input_dim, m.output_dim
    n = w.history_length
    if m.encoder_type.lower() in ('conv1d', 'cnn'):
        enc_net = NeuralNetworkDefinition(Conv1D, Conv1DConfig(
            input_channels=nu + ny, input_length=n, output_dim=nx,
            conv_channels=list(m.encoder_conv_channels),
            kernel_sizes=list(m.encoder_kernel_sizes),
            pool_sizes=list(m.encoder_pool_sizes),
            hidden_layers=list(m.encoder_hidden_layers),
            activation=m.encoder_activation, dropout=m.encoder_dropout))
    elif m.encoder_type.lower() == 'mlp':
        enc_net = NeuralNetworkDefinition(MLP, MLPConfig(
            input_dim=n * (nu + ny), output_dim=nx,
            hidden_layers=list(m.encoder_hidden_layers),
            activation=m.encoder_activation,
            final_layer_initialization_scale=m.final_layer_initialization_scale))
    else:
        raise ValueError(f'Unsupported encoder type: {m.encoder_type}')
    encoder = HistoryEncoder(HistoryEncoderConfig(enc_net, n, n, nu, ny, nx),
                             device=device, dtype=dtype, rng=generator('encoder'))
    modes = m.component_modes
    if modes['interconnection'] == 'neural':
        J = NeuralSkewMatrix(NeuralSkewMatrixConfig(mlp(nx, nx*nx, m.interconnection_hidden_layers), nx),
                             device=device, dtype=dtype, rng=generator('interconnection'))
    elif modes['interconnection'] == 'constant':
        J = ConstantSkewMatrix(ConstantSkewMatrixConfig(nx), device=device, dtype=dtype, rng=generator('interconnection'))
    else:
        raise ValueError('Unsupported interconnection mode')
    if modes['dissipation'] == 'neural':
        R = NeuralPositiveSemidefiniteMatrix(NeuralPositiveSemidefiniteMatrixConfig(
            mlp(nx, nx*m.dissipation_factor_rank, m.dissipation_hidden_layers), nx, m.dissipation_factor_rank),
            device=device, dtype=dtype, rng=generator('dissipation'))
    elif modes['dissipation'] == 'constant':
        R = ConstantPositiveSemidefiniteMatrix(ConstantPositiveSemidefiniteMatrixConfig(nx, m.dissipation_factor_rank),
                                                device=device, dtype=dtype, rng=generator('dissipation'))
    else:
        raise ValueError('Unsupported dissipation mode')
    if modes['port'] == 'neural':
        G = NeuralPortMatrix(NeuralPortMatrixConfig(mlp(nx, nx*nu, m.port_hidden_layers), nx, nu),
                             device=device, dtype=dtype, rng=generator('port'))
    elif modes['port'] == 'constant':
        G = ConstantPortMatrix(ConstantPortMatrixConfig(nx, nu), device=device, dtype=dtype, rng=generator('port'))
    else:
        raise ValueError('Unsupported port mode')
    if modes['hamiltonian'] != 'neural':
        raise ValueError('Analytical Hamiltonian requires a separately supplied module')
    if method == 'autograd':
        H = LegacyHamiltonian(LegacyHamiltonianConfig(mlp(nx, 1, m.hamiltonian_hidden_layers), m.hamiltonian_lower_bound),
                              device=device, dtype=dtype, rng=generator('hamiltonian'))
    else:
        grad_cls = {'mgradnet_m': MGradNetM, 'simple_mgradnet_m': simple_MGradNetM,
                    'gradnet_m': GradNetM}[method]
        H = GradNetHamiltonian(GradNetHamiltonianConfig(grad_cls, nx,
            m.gradnet_hidden_dim, m.gradnet_num_modules, m.gradnet_floor_eps),
            device=device, dtype=dtype, rng=generator('hamiltonian'))
    components = {'encoder': encoder, 'interconnection': J, 'dissipation': R,
                  'port': G, 'hamiltonian': H}
    for name, component in components.items():
        for parameter in component.parameters():
            parameter.requires_grad_(name in m.trainable_components)
    return PHModel(encoder=encoder, interconnection=J, dissipation=R, port=G,
                   hamiltonian=H, integrator=make_integrator(m.integrator),
                   input_dim=nu, output_dim=ny, state_dim=nx,
                   sampling_time=m.model_sampling_time or d.sampling_time,
                   history_controls=n, history_outputs=n,
                   hamiltonian_representation='autograd' if method == 'autograd' else 'gradnet',
                   integrator_substeps=m.integrator_substeps,
                   gradient_scale=m.gradient_scales[method],
                   initial_state_source='state_encoder' if m.initial_state_mode == 'encoder' else 'true_ph_state',
                   output_before_integration=d.output_before_integration)
