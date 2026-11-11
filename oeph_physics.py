"""Implement the physical reference systems and numerical integration used to generate benchmark data."""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any
import torch
import torch.nn as nn
from oeph_definitions import MSD2Config
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

class PhysicalSystem(ABC):
    """
    Abstract continuous-time physical system with sampled zero-order-hold input.

    Responsibilities
    ----------------
    A PhysicalSystem defines deterministic physics only:

        - dimensions,
        - continuous-time dynamics dx/dt = f(x, u),
        - measurement/output map y = h(x),
        - sampled rollout under a chosen one-step integrator.

    It deliberately does not define how experiments randomize initial states,
    generate controls, add noise, split trajectories or construct windows.
    """

    def __init__(
        self,
        sampling_time: float,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the component using the supplied configuration."""
        self.Ts = float(sampling_time)
        self.device = torch.device(device)
        self.dtype = dtype
        self.metadata = {} if metadata is None else dict(metadata)

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimension of the physical state x."""
        raise NotImplementedError

    @property
    @abstractmethod
    def input_dim(self) -> int:
        """Dimension of the sampled input u."""
        raise NotImplementedError

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimension of the measured output y."""
        raise NotImplementedError

    @abstractmethod
    def dynamics(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate continuous-time dynamics dx/dt = f(x, u).

        Parameters
        ----------
        x
            State batch [B, state_dim].

        u
            Input batch [B, input_dim].

        Returns
        -------
        torch.Tensor
            State derivative batch [B, state_dim].
        """
        raise NotImplementedError

    @abstractmethod
    def measurement(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate the deterministic output map y = h(x).

        Parameters
        ----------
        x
            State batch [B, state_dim].

        Returns
        -------
        torch.Tensor
            Output batch [B, output_dim].
        """
        raise NotImplementedError

    @abstractmethod
    def sample_initial_states(
        self,
        num_series: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """
        Sample physically valid initial states.

        This allows each physical system to define its own state
        ordering and sampling strategy.
        """
        raise NotImplementedError

    def simulate(
        self,
        x0: torch.Tensor,
        controls: torch.Tensor,
        *,
        integrator: str | BaseIntegrator = "rk8",
        substeps: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Roll out the physical system under sampled zero-order-hold controls.

        Parameters
        ----------
        x0
            Initial states [B, state_dim].

        controls
            Control trajectories [B, T, input_dim] or [B, T] when input_dim=1.

        integrator
            One-step integration method. Supported string values are those
            accepted by integrators.make_integrator(), including "rk4" and
            "rk8".

        substeps
            Number of fixed integration substeps per sampling interval. The
            input is held constant over all substeps in one interval.

        Returns
        -------
        states, outputs
            State and output trajectories with shapes [B, T, state_dim] and
            [B, T, output_dim].

        Sampling convention
        -------------------
        At sample k, the returned output is y_k = h(x_k). The state is then
        advanced over one sampling interval using u_k. This preserves the
        convention used in the legacy 2MSD benchmark code.
        """

        if substeps <= 0:
            raise ValueError("substeps must be positive.")

        x = self._prepare_state_batch(x0)
        u_seq = self._prepare_control_batch(controls)

        if x.shape[0] != u_seq.shape[0]:
            raise ValueError(
                "x0 and controls must have the same batch dimension. "
                f"Got x0 batch {x.shape[0]} and controls batch {u_seq.shape[0]}."
            )

        stepper = make_integrator(integrator) if isinstance(integrator, str) else integrator
        dt = self.Ts / int(substeps)

        state_samples: list[torch.Tensor] = []
        output_samples: list[torch.Tensor] = []

        for sample_index in range(u_seq.shape[1]):
            output = self.measurement(x)

            state_samples.append(x.unsqueeze(1))
            output_samples.append(output.unsqueeze(1))

            u_k = u_seq[:, sample_index, :]

            for substep_index in range(int(substeps)):
                """if sample_index > 0:
                    diagnostic_trajectory = 251

                    x_diag = x[diagnostic_trajectory]

                    if torch.max(torch.abs(x_diag)) > 100.0:
                        raise RuntimeError(
                            "\n"
                            "=========================================================\n"
                            "PHYSICAL STATE GROWTH DETECTED\n"
                            "=========================================================\n"
                            f"sample_index    : {sample_index}\n"
                            f"time [s]        : {sample_index * self.Ts:.6f}\n"
                            f"trajectory      : {diagnostic_trajectory}\n"
                            f"x               : {x_diag.cpu().numpy()}\n"
                            f"u               : {u_k[diagnostic_trajectory].cpu().numpy()}\n"
                        )"""
                x_next = stepper.step(
                    self.dynamics,
                    x,
                    u_k,
                    dt,
                )

                if not torch.isfinite(x_next).all():
                    idx = torch.nonzero(
                        ~torch.isfinite(x_next),
                        as_tuple=False,
                    )[0]

                    trajectory = idx[0].item()
                    state = idx[1].item()

                    raise RuntimeError(
                        "\n"
                        "=========================================================\n"
                        "NON-FINITE STATE DURING PHYSICAL SIMULATION\n"
                        "=========================================================\n"
                        f"sample_index    : {sample_index}\n"
                        f"substep_index   : {substep_index}\n"
                        f"time [s]        : {(sample_index + 1) * self.Ts:.6f}\n"
                        f"trajectory      : {trajectory}\n"
                        f"state           : {state}\n"
                        f"\n"
                        f"x_before        : {x[trajectory].cpu().numpy()}\n"
                        f"u               : {u_k[trajectory].cpu().numpy()}\n"
                        f"x_after         : {x_next[trajectory].cpu().numpy()}\n"
                    )

                x = x_next

        states = torch.cat(state_samples, dim=1)
        outputs = torch.cat(output_samples, dim=1)

        return states, outputs

    def system_metadata(self) -> dict[str, Any]:
        """
        Return serializable metadata describing this physical system.

        Returns
        -------
        dict
            Metadata suitable for storing inside TimeSeriesDataset.metadata.
        """

        return {
            "system_class": self.__class__.__name__,
            "sampling_time": self.Ts,
            "state_dim": self.state_dim,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            **self.metadata,
        }

    def _prepare_state_batch(
        self,
        x0: torch.Tensor,
    ) -> torch.Tensor:
        """Validate and move an initial-state batch to the system device."""

        x = torch.as_tensor(x0, dtype=self.dtype, device=self.device)

        if x.dim() == 1:
            x = x.unsqueeze(0)

        if x.dim() != 2 or x.shape[1] != self.state_dim:
            raise ValueError(
                f"Expected x0 with shape [B, {self.state_dim}], got {tuple(x.shape)}."
            )

        return x.clone()

    def _prepare_control_batch(
        self,
        controls: torch.Tensor,
    ) -> torch.Tensor:
        """Validate and move a control trajectory batch to the system device."""

        u = torch.as_tensor(controls, dtype=self.dtype, device=self.device)

        if u.dim() == 2 and self.input_dim == 1:
            u = u.unsqueeze(-1)

        if u.dim() != 3 or u.shape[2] != self.input_dim:
            raise ValueError(
                "Expected controls with shape "
                f"[B, T, {self.input_dim}], got {tuple(u.shape)}."
            )

        return u.clone()

class MSD2(PhysicalSystem):
    def __init__(
        self,
        config: MSD2Config,
        sampling_time: float = 0.1,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the component using the supplied configuration."""
        system_metadata = {
            "benchmark": "2MSD",
            "system_family": "two_mass_spring_damper",
            "physical_parameters": config.to_metadata(),
            "physical_realization": {
                "nonlinear_spring_1": bool(config.k_c1 != 0.0),
                "nonlinear_spring_2": bool(config.k_c2 != 0.0),
                "nonlinear_damper_1": bool(config.c_n1 != 0.0),
                "nonlinear_damper_2": bool(config.c_n2 != 0.0),
                "wall_contact": bool(config.K_w != 0.0 or config.chi_w != 0.0),
                "mass_contact": bool(config.K_12 != 0.0 or config.chi_12 != 0.0),
            },
            "el_state_order": ["q1", "q2", "qdot1", "qdot2"],
            "ph_state_order": ["q1", "q2", "p1", "p2"],
            "input_description": "external force applied to second mass",
            "output_description": "velocity of second mass qdot2",
        }

        if metadata is not None:
            system_metadata.update(metadata)

        super().__init__(
            sampling_time=sampling_time,
            device=device,
            dtype=dtype,
            metadata=system_metadata,
        )

        self.config = config

        self.m1 = float(config.m1)
        self.m2 = float(config.m2)

        self.L1 = float(config.L1)
        self.L2 = float(config.L2)

        self.d_w = float(config.d_w)
        self.d_12 = float(config.d_12)

        self.k_01 = float(config.k_01)
        self.k_c1 = float(config.k_c1)
        self.k_02 = float(config.k_02)
        self.k_c2 = float(config.k_c2)

        self.c_01 = float(config.c_01)
        self.c_n1 = float(config.c_n1)
        self.c_02 = float(config.c_02)
        self.c_n2 = float(config.c_n2)

        self.K_w = float(config.K_w)
        self.chi_w = float(config.chi_w)
        self.K_12 = float(config.K_12)
        self.chi_12 = float(config.chi_12)
        self.contact_exponent = float(config.contact_exponent)

    @property
    def state_dim(self) -> int:
        """Handle state dim for the experiment workflow."""
        return 4

    @property
    def input_dim(self) -> int:
        """Handle input dim for the experiment workflow."""
        return 1

    @property
    def output_dim(self) -> int:
        """Handle output dim for the experiment workflow."""
        return 1

    def sample_initial_states(
        self,
        num_series: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """
        Sample Euler-Lagrange initial states [q1, q2, qdot1, qdot2].

        The configured position interval is a displacement interval around the
        spring-equilibrium geometry. Independent displacements z1 and z2 are
        mapped to absolute particle-centre positions as

            q1 = L1 + z1,
            q2 = L1 + L2 + z2.
        """

        low_position, high_position = self.config.init_range_pos
        low_velocity, high_velocity = self.config.init_range_vel

        displacements = torch.empty(
            num_series, 2, device=self.device, dtype=self.dtype
        ).uniform_(low_position, high_position, generator=generator)

        positions = displacements.clone()
        positions[:, 0] = self.L1 + displacements[:, 0]
        positions[:, 1] = self.L1 + self.L2 + displacements[:, 1]

        velocities = torch.empty(
            num_series, 2, device=self.device, dtype=self.dtype
        ).uniform_(low_velocity, high_velocity, generator=generator)

        return torch.cat([positions, velocities], dim=1)

    def dynamics(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the complete Hunt--Crossley Euler--Lagrange dynamics."""

        if x.dim() != 2 or x.shape[1] != self.state_dim:
            raise ValueError(
                f"Expected x with shape [B, {self.state_dim}], got {tuple(x.shape)}."
            )

        q1 = x[:, 0]
        q2 = x[:, 1]
        qdot1 = x[:, 2]
        qdot2 = x[:, 3]

        u_flat = u.to(device=x.device, dtype=x.dtype).reshape(-1)
        if u_flat.shape[0] != x.shape[0]:
            raise ValueError(
                f"Input batch size {u_flat.shape[0]} does not match "
                f"state batch size {x.shape[0]}."
            )

        spring_1_compression = self.L1 - q1
        spring_2_compression = self.L2 - (q2 - q1)
        relative_velocity = qdot2 - qdot1

        wall_penetration = torch.clamp(self.d_w - q1, min=0.0)
        mass_penetration = torch.clamp(self.d_12 - (q2 - q1), min=0.0)

        spring_1_force = (
            self.k_01 * spring_1_compression
            + self.k_c1 * spring_1_compression**3
        )
        spring_2_force = (
            self.k_02 * spring_2_compression
            + self.k_c2 * spring_2_compression**3
        )

        damper_1_force = self.c_01 * qdot1 + self.c_n1 * qdot1**3
        damper_2_force = (
            self.c_02 * relative_velocity
            + self.c_n2 * relative_velocity**3
        )

        wall_contact_elastic = self.K_w * wall_penetration**self.contact_exponent
        wall_contact_dissipative = (
            self.chi_w
            * wall_penetration**self.contact_exponent
            * qdot1
        )

        mass_contact_elastic = self.K_12 * mass_penetration**self.contact_exponent
        mass_contact_dissipative = (
            self.chi_12
            * mass_penetration**self.contact_exponent
            * relative_velocity
        )

        qddot1 = (
            + spring_1_force
            - spring_2_force
            - damper_1_force
            + damper_2_force
            + wall_contact_elastic
            - wall_contact_dissipative
            - mass_contact_elastic
            + mass_contact_dissipative
        ) / self.m1

        qddot2 = (
            + spring_2_force
            - damper_2_force
            + mass_contact_elastic
            - mass_contact_dissipative
            + u_flat
        ) / self.m2

        return torch.stack([qdot1, qdot2, qddot1, qddot2], dim=1)

    def measurement(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the collocated benchmark output y = qdot2."""

        if x.dim() != 2 or x.shape[1] != self.state_dim:
            raise ValueError(
                f"Expected x with shape [B, {self.state_dim}], got {tuple(x.shape)}."
            )

        return x[:, 3:4].clone()

    def el_to_ph_state(
        self,
        x_el: torch.Tensor,
    ) -> torch.Tensor:
        """Transform [..., q1, q2, qdot1, qdot2] to [..., q1, q2, p1, p2]."""

        x = torch.as_tensor(x_el, dtype=self.dtype, device=self.device)

        if x.ndim < 1 or x.shape[-1] != self.state_dim:
            raise ValueError(
                f"Expected x_el with final dimension {self.state_dim}, "
                f"got shape {tuple(x.shape)}."
            )

        return torch.stack(
            [
                x[..., 0],
                x[..., 1],
                self.m1 * x[..., 2],
                self.m2 * x[..., 3],
            ],
            dim=-1,
        )

