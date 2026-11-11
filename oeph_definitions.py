"""Define validated configuration, dataset, model, and result structures shared by experiments."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import torch.nn as nn

METHODS = ("autograd", "mgradnet_m", "simple_mgradnet_m", "gradnet_m")
STUDIES = ("baseline", "prediction_horizon", "training_data_size", "snr", "reconstruct_results")


@dataclass
class NormalizationConfig:
    enabled: bool = True
    normalize_input: bool = True
    normalize_output: bool = True
    statistics_source: str = "training"  # train split only
    epsilon: float = 1e-8
    inverse_transform_predictions: bool = True
    inverse_transform_targets: bool = True
    physical_unit_metrics: bool = True
    nrmse_denominator: str = "full_record"


@dataclass
class DataConfig:
    system: str = "MSD"
    data_root: Path = Path("data")
    dataset_filename: str | None = None
    generation_mode: str = "continuous"
    data_seed: int = 123
    split_seed: int = 456
    train_trajectories: int = 200
    active_train_trajectories: int = 20
    validation_trajectories: int = 8
    test_trajectories: int = 20
    trajectory_length: int = 1000
    sampling_time: float = 0.1
    simulation_integrator: str = "rk4"
    simulation_substeps: int = 1
    simulation_dtype: str = "float32"
    simulation_device: str = "cpu"
    initial_position_range: tuple[float, float] = (-1.0, 1.0)
    initial_velocity_range: tuple[float, float] = (-0.5, 0.5)
    control_base_frequency: float = 0.01
    control_num_harmonics: int = 100
    control_amplitude: float = 0.5
    control_zero_probability: float = 0.0
    control_zero_interval_durations: tuple[float, ...] = (0.1, 0.15, 0.25, 0.5)
    control_phase_segment_length: int = 1000
    continuous_train_segments: int = 20
    continuous_validation_segments: int = 8
    continuous_test_segments: int = 20
    control_scale_input: bool = True
    physical_parameters: dict[str, float] = field(default_factory=dict)
    source_alignment: str = "simultaneous"
    data_alignment: str = "causal"
    output_before_integration: bool = True
    train_columns: tuple[str, str] = ("uEst", "yEst")
    validation_columns: tuple[str, str] = ("uVal", "yVal")
    sampling_time_column: str = "Ts"
    validation_raw_samples: int | None = None
    noise_snr_train: float | None = 50.0
    noise_snr_validation: float | None = 50.0
    noise_snr_test: float | None = None
    noise_seed_train: int = 1000
    noise_seed_validation: int = 1001
    noise_seed_test: int = 1002
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)


@dataclass
class ModelConfig:
    state_dim: int = 4
    input_dim: int = 1
    output_dim: int = 1
    dtype: str = "float32"
    device: str = "cpu"
    encoder_type: str = "conv1d"
    encoder_hidden_layers: tuple[int, ...] = ()
    encoder_conv_channels: tuple[int, ...] = (16, 32, 64)
    encoder_kernel_sizes: tuple[int, ...] = (37, 5, 5)
    encoder_pool_sizes: tuple[int, ...] = (2, 2, 1)
    encoder_activation: str = "tanh"
    encoder_dropout: float = 0.0
    interconnection_hidden_layers: tuple[int, ...] = (8,)
    dissipation_hidden_layers: tuple[int, ...] = (16,)
    port_hidden_layers: tuple[int, ...] = (8,)
    hamiltonian_hidden_layers: tuple[int, ...] = (16, 16)
    activation: str = "tanh"
    hamiltonian_lower_bound: float = 0.0
    final_layer_initialization_scale: float = 1.0
    gradnet_hidden_dim: int = 4
    gradnet_num_modules: int = 4
    gradnet_floor_eps: float = 1e-6
    gradient_scales: dict[str, float] = field(default_factory=lambda: {m: 1.0 for m in METHODS})
    integrator: str = "rk4"
    integrator_substeps: int = 1
    model_sampling_time: float | None = None
    initial_state_mode: str = "encoder"
    component_modes: dict[str, str] = field(default_factory=lambda: {k: "neural" for k in ("hamiltonian", "interconnection", "dissipation", "port")})
    trainable_components: tuple[str, ...] = ("encoder", "interconnection", "dissipation", "port", "hamiltonian")
    dissipation_factor_rank: int = 2
    component_seed_offsets: dict[str, int] = field(default_factory=lambda: {"encoder": 18, "interconnection": 12, "dissipation": 11, "port": 13, "autograd_hamiltonian": 14, "gradnet_hamiltonian": 15})


@dataclass
class WindowConfig:
    history_length: int = 20
    prediction_horizon: int = 200
    stride: int = 1
    train_fraction: float = 1.0
    validation_fraction: float = 1.0
    test_fraction: float = 1.0
    selection_seed: int = 0


@dataclass
class OptimizerConfig:
    name: str = "Adam"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    amsgrad: bool = False
    learning_rate_schedule: dict[int, float] | None = None
    scheduler_type: str | None = None
    scheduler_parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    epochs: int = 70
    batch_size: int = 256
    shuffle: bool = True
    shuffle_seed: int = 0
    shuffle_seed_policy: str = "fixed"
    num_workers: int = 0
    gradient_clip_norm: float | None = None
    early_stopping_patience: int | None = None
    validation_mode: str = "windowed"
    validation_full_rollout_weight: float = 0.0
    validation_nrmse_denominator: str = "full_record"
    validation_physical_metrics: bool = False
    validation_skip_initial_time: float = 0.0
    test_skip_initial_time: float = 0.0
    test_segment_length_time: float | None = None
    drop_last: bool = False
    fit_phases: tuple[int, ...] | None = None
    prediction_horizon_schedule: dict[int, int] | None = None
    state_derivative_limit_schedule: dict[int, float] | None = None
    phase_shuffle_master_seed: int | None = None
    restore_history_between_phases: bool = False
    reload_best_at_phase_boundary: bool = False
    reshuffle_at_phase_boundary: bool = False
    validate_initial: bool = False
    square_outputs: bool = False
    optimizers: dict[str, OptimizerConfig] = field(default_factory=dict)


@dataclass
class StudyConfig:
    study_type: str = "baseline"
    methods: tuple[str, ...] = ("autograd", "mgradnet_m")
    method_preprocessing: dict[str, dict[str, Any]] = field(default_factory=dict)
    seed_mode: str = "same_seed_all_components"
    manual_component_seeds: dict[str, int] = field(default_factory=dict)
    seed_master: int = 123
    num_seeds: int = 10
    explicit_model_seeds: tuple[int, ...] | None = None
    selected_seeds: dict[str, int] = field(default_factory=dict)
    num_shuffles: int = 1
    shuffle_seed_master: int = 667
    explicit_shuffle_seeds: tuple[int, ...] | None = None
    horizon_levels: tuple[int, ...] = (10, 20, 40, 100, 200)
    data_size_levels: tuple[int, ...] = (10000,)
    snr_levels: tuple[float, ...] = (5.0, 15.0, 35.0, 50.0)
    results_root: Path = Path("results")
    study_name: str = "streamlined"
    reconstruction_directory: Path | None = None
    save_models: bool = True
    save_predictions: bool = True
    save_figures: bool = True
    save_statistics: bool = True
    confidence_level: float = 0.95
    plot_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    windows: WindowConfig
    training: TrainingConfig
    study: StudyConfig


@dataclass
class RunResult:
    method: str
    model_seed: int
    shuffle_seed: int
    configuration: dict[str, Any]
    training_loss: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    checkpoint_path: str | None = None
    predictions_path: str | None = None
    normalization_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StudyResult:
    configuration: ExperimentConfig
    runs: list[RunResult] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)
    figures: list[str] = field(default_factory=list)


def validate_configuration(config: ExperimentConfig) -> None:
    """Validate configuration structure without executing an experiment."""
    d, m, w, t, s = config.data, config.model, config.windows, config.training, config.study
    if d.system not in ("MSD", "CT"):
        raise ValueError(f"Unknown system: {d.system}")
    if (not isinstance(s.study_name, str) or not s.study_name.strip()
            or s.study_name in (".", "..")
            or "/" in s.study_name or "\\" in s.study_name):
        raise ValueError("EXPERIMENT_TITLE must be a nonempty single folder name")
    if s.study_type not in STUDIES:
        raise ValueError(f"Unknown study: {s.study_type}")
    if not s.methods or len(set(s.methods)) != len(s.methods) or any(x not in METHODS for x in s.methods):
        raise ValueError(f"Methods must be a nonempty unique subset of {METHODS}")
    if min(m.state_dim, m.input_dim, m.output_dim, w.history_length, w.prediction_horizon, w.stride, t.batch_size) <= 0:
        raise ValueError("Dimensions, history, horizon, stride and batch size must be positive")
    if t.epochs <= 0 or s.num_seeds <= 0 or s.num_shuffles <= 0:
        raise ValueError("Epoch and seed counts must be positive")
    if s.explicit_model_seeds is not None and (not s.explicit_model_seeds or any(int(x) < 0 for x in s.explicit_model_seeds)):
        raise ValueError("EXPLICIT_MODEL_SEEDS must contain nonnegative integers")
    if s.explicit_shuffle_seeds is not None and (not s.explicit_shuffle_seeds or any(int(x) < 0 for x in s.explicit_shuffle_seeds)):
        raise ValueError("EXPLICIT_SHUFFLE_SEEDS must contain nonnegative integers")
    if any(not 0 < f <= 1 for f in (w.train_fraction, w.validation_fraction, w.test_fraction)):
        raise ValueError("Window fractions must lie in (0, 1]")
    if d.normalization.epsilon <= 0:
        raise ValueError("Normalization epsilon must be positive")
    if d.source_alignment not in ("simultaneous", "causal") or d.data_alignment not in ("simultaneous", "causal"):
        raise ValueError("Unknown data alignment")
    if s.study_type == "reconstruct_results" and s.reconstruction_directory is None:
        raise ValueError("Specify reconstruction_directory for reconstruct_results")
    if s.study_type == "prediction_horizon" and not s.horizon_levels:
        raise ValueError("Empty horizon levels")
    if s.study_type == "training_data_size" and not s.data_size_levels:
        raise ValueError("Empty data-size levels")
    if s.study_type == "snr" and not s.snr_levels:
        raise ValueError("Empty SNR levels")
    if d.system == "CT" and not d.dataset_filename:
        raise ValueError("CT requires a dataset filename")
    allowed_preprocessing = {'normalize', 'normalize_input', 'normalize_output',
                             'statistics_source', 'epsilon', 'square_outputs',
                             'physical_unit_metrics', 'inverse_transform_predictions',
                             'inverse_transform_targets', 'nrmse_denominator'}
    for method, overrides in s.method_preprocessing.items():
        if method not in s.methods:
            raise ValueError(f'Preprocessing override for unselected method {method!r}')
        unknown = set(overrides) - allowed_preprocessing
        if unknown:
            raise ValueError(f'Unknown preprocessing options for {method}: {sorted(unknown)}')
    for method in s.methods:
        if method not in t.optimizers:
            raise ValueError(f"Missing optimizer for {method}")
        opt = t.optimizers[method]
        if opt.learning_rate <= 0 or (opt.learning_rate_schedule and opt.scheduler_type):
            raise ValueError(f"Invalid learning-rate configuration for {method}")





# Stage 2: data and physics containers (one location for all containers).
from typing import Optional
import numpy as np
import torch

@dataclass
class TimeSeriesDataset:
    """
    Canonical representation of a collection of aligned time series.

    Shapes
    ------
    controls
        [I, T, input_dim]

    outputs
        [I, T, output_dim]

    states
        [I, T, state_dim]
        Optional simulator-native physical state trajectories.

    states_ph
        [I, T, ph_state_dim]
        Optional canonical port-Hamiltonian state trajectories.

    time
        [I, T]

    sample_index
        [I, T]

    Parameters
    ----------
    controls
        Input time series.

    outputs
        Output time series.

    states
        Optional simulator-native physical state trajectories.

    states_ph
        Optional canonical port-Hamiltonian state trajectories.

    time
        Time vector for every sample.

    sample_index
        Original sample index corresponding to every sample.

    Ts
        Sampling period.
        May be None for non-uniformly sampled data.

    series_ids
        Integer identifier for each trajectory.

    metadata
        Arbitrary experiment metadata.
    """

    controls: np.ndarray

    outputs: np.ndarray

    states: Optional[np.ndarray]

    states_ph: Optional[np.ndarray]

    time: np.ndarray

    sample_index: np.ndarray

    Ts: Optional[float]

    series_ids: np.ndarray

    metadata: dict

    @property
    def num_series(self):
        """Handle num series for the experiment workflow. Used to validate or expose experiment configuration and data dimensions."""
        return self.controls.shape[0]

    @property
    def series_length(self):
        """Handle series length for the experiment workflow. Used to validate or expose experiment configuration and data dimensions."""
        return self.controls.shape[1]

    @property
    def input_dim(self):
        """Handle input dim for the experiment workflow. Used to validate or expose experiment configuration and data dimensions."""
        return self.controls.shape[2]

    @property
    def output_dim(self):
        """Handle output dim for the experiment workflow. Used to validate or expose experiment configuration and data dimensions."""
        return self.outputs.shape[2]

    @property
    def state_dim(self):
        """Handle state dim for the experiment workflow. Used to validate or expose experiment configuration and data dimensions."""
        if self.states is None:
            return None
        return self.states.shape[2]
    
    @property
    def ph_state_dim(self):
        """Handle ph state dim for the experiment workflow. Used to validate or expose experiment configuration and data dimensions."""
        if self.states_ph is None:
            return None
        return self.states_ph.shape[2]

@dataclass
class SplitDataset:
    """
    Canonical train/validation/test split.

    Each split is itself a TimeSeriesDataset.
    """

    train: TimeSeriesDataset

    validation: TimeSeriesDataset

    test: TimeSeriesDataset

@dataclass(frozen=True)
class MSD2Config:
    """
    Complete physical definition of the two-mass-spring-damper benchmark.

    The configuration represents the most general Hunt--Crossley 2MSD model.
    Simpler benchmark realizations are obtained only by setting the desired
    constitutive coefficients to zero; the mechanical topology is unchanged.

    All benchmark-specific numerical values must be supplied explicitly by the
    runner. The library does not define hidden physical benchmark constants.

    Initial-position ranges are interpreted as displacement ranges around the
    spring-equilibrium configuration. For sampled displacements z1 and z2,

        q1 = L1 + z1
        q2 = L1 + L2 + z2.

    This preserves the benchmark's displacement-based initial-condition API
    while the physical state uses the absolute particle-centre coordinates of
    the analytical Hunt--Crossley derivation.
    """

    m1: float
    m2: float

    L1: float
    L2: float

    d_w: float
    d_12: float

    k_01: float
    k_c1: float
    k_02: float
    k_c2: float

    c_01: float
    c_n1: float
    c_02: float
    c_n2: float

    K_w: float
    chi_w: float
    K_12: float
    chi_12: float
    contact_exponent: float

    init_range_pos: tuple[float, float] = (-1.0, 1.0)
    init_range_vel: tuple[float, float] = (-0.5, 0.5)

    def to_metadata(self) -> dict[str, Any]:
        """Return a plain-Python metadata representation."""
        return asdict(self)

@dataclass(frozen=True)
class InitialConditionConfig:
    """
    Initial-state sampling configuration for simulated trajectory generation.

    The default values reproduce the legacy 2MSD benchmark ranges.
    """

    position_range: tuple[float, float] = (-1.0, 1.0)
    velocity_range: tuple[float, float] = (-0.5, 0.5)
    distribution: str = "uniform"
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ControlConfig:
    """
    Multi-sine control-generation configuration.

    Parameters mirror the legacy benchmark control generator:

        u_b(t_k) = sum_r sin(2*pi*r*f0*t_k + phi_{b,r})

    with optional contiguous zero/active regions and post-mask RMS scaling.
    """

    base_frequency: float = 0.01
    num_harmonics: int = 100
    zero_probability: float = 0.0
    zero_interval_durations: tuple[float, ...] = (0.1, 0.5, 1.0, 2.0, 4.0)
    max_contiguous_interval_duration: float | None = None
    scale_input: bool = False
    amplitude: float = 0.1
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class NoiseConfig:
    """
    Additive white-noise configuration for generated datasets.

    Parameters
    ----------
    control_snr_db
        Requested SNR for noisy controls. If None, controls remain clean.

    output_snr_db
        Requested SNR for noisy outputs. If None, outputs remain clean.

    add_control_noise
        Whether to corrupt controls.

    add_output_noise
        Whether to corrupt outputs.

    center_noise
        If True, remove each trajectory/channel noise mean before scaling.
        This preserves the legacy measurement-noise convention.
    """

    control_snr_db: float | None = None
    output_snr_db: float | None = None
    add_control_noise: bool = False
    add_output_noise: bool = True
    center_noise: bool = True
    eps: float = 1e-12
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class SimulationConfig:
    """
    Complete configuration for one simulated TimeSeriesDataset.

    This is the Stage-0 data-source configuration that replaces the old dummy
    dataset generator in user-facing runners.
    """

    num_series: int
    series_length: int
    sampling_time: float
    integrator: str = "rk8"
    integrator_substeps: int = 1
    seed: int = 0
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    source_name: str = "simulated_physical_system"
    control_phase_segment_length: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# Stage 3: central neural-component configuration containers
@dataclass
class MLPConfig:
    """
    Canonical configuration for feedforward neural networks.

    Parameters
    ----------
    input_dim
        Input dimension.

    output_dim
        Output dimension.

    hidden_layers
        Width of each hidden layer.

    activation
        Hidden-layer activation name.

    lower_bound
        Optional lower-bound shift for the network output.

    final_layer_initialization_scale
        Multiplicative scale applied to the Xavier-initialized weights and
        biases of the final linear layer only. Hidden-layer initialization
        is unchanged.
    """

    input_dim: int
    output_dim: int
    hidden_layers: list[int] = field(default_factory=lambda: [10])
    activation: str = "tanh"
    lower_bound: Optional[float] = None
    final_layer_initialization_scale: float = 1.0


@dataclass
class Conv1DConfig:
    """
    Canonical configuration for one-dimensional convolutional neural networks.

    Parameters
    ----------
    input_channels
        Number of input channels.

    input_length
        Length of the one-dimensional input axis.

    output_dim
        Output dimension.

    conv_channels
        Output-channel count of each convolutional layer.

    kernel_sizes
        Kernel size of each convolutional layer.

    pool_sizes
        Max-pooling size following each convolutional layer. A value of one
        disables pooling for that layer.

    hidden_layers
        Widths of the MLP head following the convolutional stack. An empty
        list creates a single linear projection from the flattened convolutional
        output directly to ``output_dim``.

    activation
        Activation used after convolutional and hidden linear layers.
    """

    input_channels: int
    input_length: int
    output_dim: int
    conv_channels: list[int] = field(default_factory=lambda: [32, 64])
    kernel_sizes: list[int] = field(default_factory=lambda: [5, 5])
    pool_sizes: list[int] = field(default_factory=lambda: [1, 1])
    hidden_layers: list[int] = field(default_factory=lambda: [128])
    activation: str = "relu"
    dropout: float = 0.0


@dataclass
class GradNetConfig:
    """
    Configuration object for GradNet construction.

    The GradNet classes themselves are not refactored. This dataclass is only a
    lightweight serializable specification that model factories can unpack when
    instantiating one of the fixed GradNet implementations.
    """

    input_dim: int
    hidden_dim: int = 4
    M: int = 4
    floor_eps: float = 1e-6


def _require_architecture_dimension(*args, **kwargs):
    """Validate architecture dimension against the required model dimensions. Used to validate or expose experiment configuration and data dimensions."""
    from oeph_neural_models import _require_architecture_dimension as check
    return check(*args, **kwargs)

def _validate_optional_shape(*args, **kwargs):
    """Validate optional shape against the required model dimensions. Used to validate or expose experiment configuration and data dimensions."""
    from oeph_neural_models import _validate_optional_shape as check
    return check(*args, **kwargs)
@dataclass(frozen=True)
class GradNetHamiltonianConfig:
    """Definition of one GradNet Hamiltonian."""

    gradnet_type: type[nn.Module]

    input_dim: int

    hidden_dim: int

    num_modules: int

    floor_eps: float = 1e-6

    grad_scale: float = 1.0

    def __post_init__(self):

        """Validate and finalize dataclass fields after initialization."""
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        if self.num_modules <= 0:
            raise ValueError("num_modules must be positive.")


@dataclass(frozen=True)
class LegacyHamiltonianConfig:

    network: NeuralNetworkDefinition

    lower_bound: float = 0.0


@dataclass(frozen=True)
class NeuralNetworkDefinition:
    """Definition of one generic neural network used inside a PH component."""

    network_type: type[nn.Module]
    network_architecture: Any

    def __post_init__(self) -> None:
        """Validate and finalize dataclass fields after initialization."""
        if not isinstance(self.network_type, type):
            raise TypeError("network_type must be a torch.nn.Module class.")
        if not issubclass(self.network_type, nn.Module):
            raise TypeError("network_type must inherit from torch.nn.Module.")


@dataclass(frozen=True)
class HistoryEncoderConfig:
    """Scientific definition of a history-based latent-state encoder."""

    network: NeuralNetworkDefinition
    history_controls: int
    history_outputs: int
    input_dim: int
    output_dim: int
    state_dim: int

    def __post_init__(self) -> None:
        """Validate and finalize dataclass fields after initialization."""
        for name in (
            "history_controls",
            "history_outputs",
            "input_dim",
            "output_dim",
            "state_dim",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")

        from oeph_neural_models import Conv1DConfig, _require_architecture_dimension
        architecture = self.network.network_architecture

        if self.network.network_type.__name__ == "Conv1D":
            if not isinstance(architecture, Conv1DConfig):
                raise TypeError(
                    "HistoryEncoder with Conv1D requires Conv1DConfig."
                )
            if self.history_controls != self.history_outputs:
                raise ValueError(
                    "HistoryEncoder with Conv1D requires equal control and output "
                    "history lengths."
                )
            _require_architecture_dimension(
                architecture,
                name="input_channels",
                expected=self.input_dim + self.output_dim,
                component="HistoryEncoder",
            )
            _require_architecture_dimension(
                architecture,
                name="input_length",
                expected=self.history_controls,
                component="HistoryEncoder",
            )
        else:
            expected_input_dim = (
                self.history_controls * self.input_dim
                + self.history_outputs * self.output_dim
            )
            _require_architecture_dimension(
                architecture,
                name="input_dim",
                expected=expected_input_dim,
                component="HistoryEncoder",
            )

        _require_architecture_dimension(
            architecture,
            name="output_dim",
            expected=self.state_dim,
            component="HistoryEncoder",
        )


@dataclass(frozen=True)
class NeuralSkewMatrixConfig:
    """Definition of a state-dependent skew-symmetric matrix J(x)."""

    network: NeuralNetworkDefinition
    state_dim: int

    def __post_init__(self) -> None:
        """Validate and finalize dataclass fields after initialization."""
        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive.")
        _require_architecture_dimension(
            self.network.network_architecture,
            name="input_dim",
            expected=self.state_dim,
            component="NeuralSkewMatrix",
        )
        _require_architecture_dimension(
            self.network.network_architecture,
            name="output_dim",
            expected=self.state_dim * self.state_dim,
            component="NeuralSkewMatrix",
        )


@dataclass(frozen=True)
class ConstantSkewMatrixConfig:
    """Definition of a state-independent skew-symmetric matrix."""

    state_dim: int
    initial_matrix: Any | None = None
    trainable: bool = True
    initialization_scale: float = 1.0

    def __post_init__(self) -> None:
        """Validate and finalize dataclass fields after initialization."""
        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive.")
        if self.initialization_scale <= 0:
            raise ValueError("initialization_scale must be positive.")
        _validate_optional_shape(self.initial_matrix, (self.state_dim, self.state_dim), "initial_matrix")


@dataclass(frozen=True)
class NeuralPositiveSemidefiniteMatrixConfig:
    """Definition of a state-dependent PSD matrix R(x)."""

    network: NeuralNetworkDefinition
    state_dim: int
    factor_rank: int | None = None

    def __post_init__(self) -> None:
        """Validate and finalize dataclass fields after initialization."""
        rank = self.state_dim if self.factor_rank is None else int(self.factor_rank)
        if self.state_dim <= 0 or rank <= 0:
            raise ValueError("state_dim and factor_rank must be positive.")
        _require_architecture_dimension(
            self.network.network_architecture,
            name="input_dim",
            expected=self.state_dim,
            component="NeuralPositiveSemidefiniteMatrix",
        )
        _require_architecture_dimension(
            self.network.network_architecture,
            name="output_dim",
            expected=self.state_dim * rank,
            component="NeuralPositiveSemidefiniteMatrix",
        )


@dataclass(frozen=True)
class ConstantPositiveSemidefiniteMatrixConfig:
    """Definition of a state-independent PSD matrix."""

    state_dim: int
    factor_rank: int | None = None
    initial_factor: Any | None = None
    trainable: bool = True
    initialization_scale: float = 1.0

    def __post_init__(self) -> None:
        """Validate and finalize dataclass fields after initialization."""
        rank = self.state_dim if self.factor_rank is None else int(self.factor_rank)
        if self.state_dim <= 0 or rank <= 0:
            raise ValueError("state_dim and factor_rank must be positive.")
        if self.initialization_scale <= 0:
            raise ValueError("initialization_scale must be positive.")
        _validate_optional_shape(self.initial_factor, (self.state_dim, rank), "initial_factor")


@dataclass(frozen=True)
class NeuralPortMatrixConfig:
    """Definition of a state-dependent port matrix G(x)."""

    network: NeuralNetworkDefinition
    state_dim: int
    input_dim: int

    def __post_init__(self) -> None:
        """Validate and finalize dataclass fields after initialization."""
        if self.state_dim <= 0 or self.input_dim <= 0:
            raise ValueError("state_dim and input_dim must be positive.")
        _require_architecture_dimension(
            self.network.network_architecture,
            name="input_dim",
            expected=self.state_dim,
            component="NeuralPortMatrix",
        )
        _require_architecture_dimension(
            self.network.network_architecture,
            name="output_dim",
            expected=self.state_dim * self.input_dim,
            component="NeuralPortMatrix",
        )


@dataclass(frozen=True)
class ConstantPortMatrixConfig:
    """Definition of a state-independent port matrix."""

    state_dim: int
    input_dim: int
    initial_matrix: Any | None = None
    trainable: bool = True
    initialization_scale: float = 1.0

    def __post_init__(self) -> None:
        """Validate and finalize dataclass fields after initialization."""
        if self.state_dim <= 0 or self.input_dim <= 0:
            raise ValueError("state_dim and input_dim must be positive.")
        if self.initialization_scale <= 0:
            raise ValueError("initialization_scale must be positive.")
        _validate_optional_shape(self.initial_matrix, (self.state_dim, self.input_dim), "initial_matrix")



# Window data containers (single canonical definitions)
@dataclass
class WindowConstructionMetadata:
    """
    Metadata describing the window construction process.
    """

    #
    # Window geometry
    #
    history_length: int
    prediction_length: int

    #
    # Window selection
    #
    stride: int = 1

    selection_fraction: float = 1.0

    sampling_method: str = "random"

    sampling_seed: int | None = None

    #
    # Statistics
    #
    possible_windows: int = 0

    strided_windows: int = 0

    selected_windows: int = 0
    sample_alignment: str = "causal"



# =============================================================================
# Window dataset
# =============================================================================

@dataclass
class WindowDataset:
    """
    Canonical representation of supervised learning windows.

    Every row corresponds to one training sample.
    """

    #
    # History
    #
    history_controls: np.ndarray
    history_outputs: np.ndarray
    history_states: np.ndarray | None
    history_states_ph: np.ndarray | None

    #
    # Prediction
    #
    future_controls: np.ndarray
    future_outputs: np.ndarray
    future_states: np.ndarray | None
    future_states_ph: np.ndarray | None

    #
    # Provenance
    #
    window_ids: np.ndarray
    series_ids: np.ndarray
    window_starts: np.ndarray

    #
    # Temporal information
    #
    history_time: np.ndarray
    future_time: np.ndarray

    history_sample_index: np.ndarray
    future_sample_index: np.ndarray

    #
    # Metadata
    #
    metadata: WindowConstructionMetadata

    @property
    def num_windows(self):
        """Handle num windows for the experiment workflow. Used to validate or expose experiment configuration and data dimensions."""
        return self.history_controls.shape[0]

    @property
    def history_length(self):
        """Handle history length for the experiment workflow. Used to validate or expose experiment configuration and data dimensions."""
        return self.history_controls.shape[1]

    @property
    def prediction_length(self):
        """Handle prediction length for the experiment workflow. Used to validate or expose experiment configuration and data dimensions."""
        return self.future_outputs.shape[1]

    @property
    def input_dim(self):
        """Handle input dim for the experiment workflow. Used to validate or expose experiment configuration and data dimensions."""
        return self.history_controls.shape[2]

    @property
    def output_dim(self):
        """Handle output dim for the experiment workflow. Used to validate or expose experiment configuration and data dimensions."""
        return self.history_outputs.shape[2]
    
    @property
    def num_series(self):
        """
        Number of original time series contributing windows.
        """
        return len(
            np.unique(
                self.series_ids,
            )
        )

    @property
    def windows_per_series(self):
        """
        Number of windows originating from each time series.
        """

        _, counts = np.unique(
            self.series_ids,
            return_counts=True,
        )

        return counts
    

@dataclass
class WindowSplitDataset:
    """
    Windowed train/validation/test datasets.
    """

    train: WindowDataset

    validation: WindowDataset

    test: WindowDataset




# Training and test results
@dataclass
class FitResult:
    model: Any
    best_model_state: dict[str, Any]
    best_validation_loss: float
    best_validation_epoch: int
    training_loss_history: list[float]
    validation_loss_history: list[float]
    learning_rate_history: list[float]
    epochs_completed: int

@dataclass
class TestResult:
    predictions: list[np.ndarray]
    targets: list[np.ndarray]
    mse: float
    rmse: float
    nrmse: float

