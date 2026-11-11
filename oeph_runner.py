"""Configure and run OE-pHNN comparisons for the MSD and cascaded-tanks systems.

QUICK START — edit the settings below, then run ``python oeph_runner.py``.

- Experiment naming: choose SYSTEM (MSD or CT), STUDY, METHODS, and
  EXPERIMENT_TITLE under "EXPERIMENT NAMING AND STUDY SELECTION".
- Repetitions and sweeps: set NUM_SEEDS / EXPLICIT_MODEL_SEEDS, seed policy,
  and the appropriate HORIZON_LEVELS, DATA_SIZE_LEVELS, or SNR_LEVELS.
- Data generation/loading: edit MSD_DATA_* simulation/source settings or
  CT_DATA_* measured-data paths under the corresponding DATA SOURCE heading.
- Data parametrisation: set noise, normalization, METHOD_PREPROCESSING,
  history length, and window sampling under DATA PREPROCESSING / WINDOWING.
- Model parametrisation: set MSD_MODEL_* or CT_MODEL_* for encoder, J, R,
  G, Hamiltonian, gradient network, integration, and initialization.
- Train/validation/test parametrisation: edit *_TRAINING_* and *_OPT_*
  for batch size, epoch blocks, schedules, validation and test evaluation.
- Figures/reports: use SAVE_FIGURES and PLOT_* in PLOTTING AND OUTPUT.
- Reconstruction: choose STUDY="reconstruct_results" and point
  RECONSTRUCTION_DIRECTORY at an existing results directory.

For the paper baseline, select SYSTEM and set EXPERIMENT_TITLE to a unique name.
- The baseline settings below use 10 model seeds and the source scripts' exact seed draws.
- MSD: 20,000 training samples, horizon 200, 70 epochs, 50 dB;
  source-script horizon warm-up 5 -> 200 at epoch 5.
- CT: history 4, horizon 60, 700 epochs, batch 32.
- Set STUDY to "snr", "training_data_size", or "prediction_horizon" for sweeps.
- CT PH-ablation AG and MGM both start at LR 0.01 with the same
  scheduled reductions; the PH archives are the configuration reference.

Only edit the configuration constants above FINAL CONFIGURATION ASSEMBLY.
The assembly and execution sections below translate these settings into
experiment objects; they are not additional user-facing configuration.
The notebook oeph_comparisons.ipynb is intended as the simpler public entry
point; this runner retains the complete explicit configuration.
"""
from __future__ import annotations
from pathlib import Path
from oeph_definitions import (
    DataConfig, ExperimentConfig, ModelConfig, NormalizationConfig,
    OptimizerConfig, StudyConfig, TrainingConfig, WindowConfig,
    validate_configuration,
)
# ===================== 1. EXPERIMENT NAMING AND STUDY SELECTION =====================
# - Available options: "MSD" (simulated mass-spring-damper), "CT" (measured cascaded tanks).
SYSTEM = "CT"                # "MSD", "CT"; paper baseline defaults to MSD
# "baseline": one configuration; 
# "prediction_horizon": vary rollout length; 
# "training_data_size": vary available training examples; 
# "snr": vary training noise; 
# "reconstruct_results": regenerate reports/plots from saved outputs without training.
# - Available options: "baseline", "prediction_horizon", "training_data_size", "snr", "reconstruct_results".
STUDY = "baseline"             # baseline, prediction_horizon, training_data_size,
                               # snr, reconstruct_results
# Any nonempty unique subset: "autograd" (autodiff Hamiltonian), "mgradnet_m" (modular gradient network), "simple_mgradnet_m" (simplified modular variant), "gradnet_m" (gradient network). Order is execution order.
# - Available options: tuple of distinct method names: "autograd", "mgradnet_m", "simple_mgradnet_m", "gradnet_m".
METHODS = ("autograd", "mgradnet_m")
# Method-specific overrides. An empty dict inherits the system defaults below.
# Supported: normalize, normalize_input, normalize_output, statistics_source,
# epsilon, square_outputs, physical_unit_metrics, inverse_transform_predictions,
# inverse_transform_targets, nrmse_denominator.
# Example: {'autograd': {'normalize': False, 'square_outputs': False},
#           'mgradnet_m': {'normalize': True, 'square_outputs': True}}
# - Available options: dictionary keyed by selected method; {} inherits system defaults; keys include normalize and square_outputs.
METHOD_PREPROCESSING = {
    'autograd': {'normalize': (SYSTEM == 'CT'), 'square_outputs': False},
    'mgradnet_m': {'normalize': False, 'square_outputs': False},
}

# Optional: "simple_mgradnet_m", "gradnet_m"
# Experiment title is the EXACT subdirectory name beneath RESULTS_ROOT.
# Example: EXPERIMENT_TITLE="MSD_horizon_3seeds" -> results/MSD_horizon_3seeds/
# Use a new title for each experiment: existing files may be overwritten.
# Use a single folder name (no slashes, backslashes, ".", or "..").
# - Available options: nonempty single directory name (string).
EXPERIMENT_TITLE = "CT_paper_baseline"  # use "CT_paper_baseline" for CT
# Parent output directory for checkpoints, datasets, metrics, reports and plots.
# - Available options: Path(...); RECONSTRUCTION_DIRECTORY also accepts None.
RESULTS_ROOT = Path("results")
# For STUDY="reconstruct_results", set Path to an existing study results directory; otherwise None.
# - Available options: Path(...); RECONSTRUCTION_DIRECTORY also accepts None.
RECONSTRUCTION_DIRECTORY = None  # Path("results/previous_run")

# STUDY chooses the SINGLE variable to vary; NUM_SEEDS independently sets
# the number of model initializations PER LEVEL, for ALL studies.
# Example: STUDY="snr", SNR_LEVELS=(10,30), NUM_SEEDS=3 executes
# 2 SNR levels x 3 model seeds x len(METHODS) runs (NUM_SHUFFLES=1).
# "baseline" also runs NUM_SEEDS model seeds at its single level.
# The same generated model seeds are reused at every level (paired study).
# All results, checkpoints, histories, statistics and plots are saved.
# Number of model seeds PER study level when EXPLICIT_MODEL_SEEDS is None.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
NUM_SEEDS = 10
# Master seed used to generate model seeds; not itself necessarily a model initialization seed.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
SEED_MASTER = 123 if SYSTEM == "MSD" else 666
# None: generate NUM_SEEDS seeds; otherwise use these exact model seeds at EVERY level.
# - Available options: None (disabled/automatic), or a compatible value described above.
# Exact model-seed draws in the original scripts (NumPy default_rng).
# MSD: COMPARISON_MASTER_SEED=123, integers(0, 2**31-1, 10).
# CT: SEED_MASTER=666, integers(0, 10_000_000, 10).
# These explicit lists avoid dependence on the refactored seed generator's range.
PAPER_MSD_MODEL_SEEDS = (33158374, 1465339467, 1273345680, 115579757, 1952249162, 473219222, 547800578, 395935448, 716435562, 377755045)
PAPER_CT_MODEL_SEEDS = (8905113, 7641621, 2540546, 5504982, 616361, 5421910, 3401864, 6139353, 4817305, 3416967)
# To use exact paper model seeds set to: 
# EXPLICIT_MODEL_SEEDS = PAPER_MSD_MODEL_SEEDS if SYSTEM == "MSD" else PAPER_CT_MODEL_SEEDS
# To use random seeds, set to None
EXPLICIT_MODEL_SEEDS = None
# Dictionary method-name -> one integer seed (overrides NUM_SEEDS for that method); {} shares the seed list across methods.
# - Available options: configuration value; see description above.
SELECTED_SEEDS = {}              # e.g. {"autograd": 123, "mgradnet_m": 123}
# Component initialization modes (independent of NUM_SEEDS and STUDY):
# "same_seed_all_components": each of Encoder/J/R/G/H uses the base model seed.
# "independent_component_seeds": derive distinct seeds from each base seed s:
#     Encoder=5*s, J=5*s+1, R=5*s+2, G=5*s+3, Hamiltonian=5*s+4.
#     The same derived seeds are reused across methods, so corresponding
#     Encoder/J/R/G initializations are paired across methods.
# "manual_seeding": use MANUAL_COMPONENT_SEEDS exactly as specified below.
#     These five seeds are FIXED across all NUM_SEEDS repetitions and methods;
#     changing the base model seed does not change the initialization.
#     For independent repetitions use "independent_component_seeds" instead.
# - Available options: "same_seed_all_components", "independent_component_seeds", "manual_seeding".
SEED_MODE = "independent_component_seeds"
# - Available options: configuration value; see description above.
MANUAL_COMPONENT_SEEDS = {
    "encoder": 100,
    "interconnection": 101,  # J
    "dissipation": 102,     # R
    "port": 103,            # G
    "hamiltonian": 104,     # H
}

# Independent shuffle repetitions PER model seed; 1 uses the reference TRAINING shuffle seed.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
NUM_SHUFFLES = 1
# Master seed used to generate the per-run shuffle seeds.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
SHUFFLE_SEED_MASTER = 667
# None: use reference shuffle seed if NUM_SHUFFLES=1; generate seeds if >1. Explicit tuple overrides.
# - Available options: None (disabled/automatic), or a compatible value described above.
EXPLICIT_SHUFFLE_SEEDS = None
# Positive integer prediction lengths used only for STUDY="prediction_horizon".
# - Available options: tuple of values with the meaning and units described above.
HORIZON_LEVELS = (20, 50, 100, 200) if SYSTEM == "MSD" else (15, 30, 60, 120)
# Positive training-data levels: MSD = active training trajectories; CT = training windows.
# - Available options: tuple of values with the meaning and units described above.
DATA_SIZE_LEVELS = (1, 2, 5, 10, 20)  # MSD: 1000 samples per segment
# Noise signal-to-noise ratios in dB for STUDY="snr"; larger values mean less noise.
# - Available options: tuple of values with the meaning and units described above.
SNR_LEVELS = (15.0, 35.0, 40.0, 50.0)


# ===================== 2. PLOTTING AND OUTPUT =====================
# Saved figures and re-plotting options. Used for every study, including ablations.
# PLOT_SCALES: "linear", "log", or both; PLOT_FORMATS: "png", "svg", or both.
# PLOT_TRAINING_METRIC: "mse" or "rmse" (per-epoch histories are MSE).
# PLOT_TEST_METRIC: "mse", "rmse", or "nrmse" (evaluated on test predictions).
# PLOT_AGGREGATE: True overlays seed means and confidence intervals.
# To re-plot without training: STUDY="reconstruct_results" and set
# RECONSTRUCTION_DIRECTORY to the saved experiment directory.
# - Available options: True (enable) or False (disable).
SAVE_FIGURES = True
# - Available options: tuple drawn from "linear", "log".
PLOT_SCALES = ("linear", "log")
# - Available options: tuple drawn from "png", "svg".
PLOT_FORMATS = ("svg",)
# - Available options: "mse", "rmse".
PLOT_TRAINING_METRIC = "mse"
# - Available options: "mse", "rmse", "nrmse".
PLOT_TEST_METRIC = "nrmse"
# - Available options: True (enable) or False (disable).
PLOT_AGGREGATE = True

# Paper-figure presentation (adapted from reconstruct_paper_figures.py).
# - Available options: True (enable) or False (disable).
PLOT_MAKE_INDIVIDUAL = True
# - Available options: True (enable) or False (disable).
PLOT_MAKE_COMBINED = True
# - Available options: True (enable) or False (disable).
PLOT_MAKE_MEAN_CI = True
# - Available options: True (enable) or False (disable).
PLOT_MAKE_CONVERGENCE = True
# - Available options: True (enable) or False (disable).
PLOT_SHOW_INDIVIDUAL_RUNS = True
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
PLOT_INDIVIDUAL_RUN_ALPHA = 0.40
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
PLOT_CONFIDENCE = 0.95
# - Available options: tuple of values with the meaning and units described above.
PLOT_TRAIN_FIGURE_SHAPE = (12, 5)
# - Available options: tuple of values with the meaning and units described above.
PLOT_ROLLOUT_FIGURE_SHAPE = (10, 5)
# - Available options: None (disabled/automatic), or a compatible value described above.
PLOT_TRAIN_TITLE = None
# - Available options: None (disabled/automatic), or a compatible value described above.
PLOT_ROLLOUT_TITLE = None
# - Available options: configuration value; see description above.
PLOT_TRAIN_AXIS_LABELS = {"MSD": {"x": "Epoch", "train_y": None, "validation_y": None},
                          "CT": {"x": "Epoch", "train_y": None, "validation_y": None}}
# - Available options: configuration value; see description above.
PLOT_ROLLOUT_AXIS_LABELS = {"MSD": {"x": "Time (s)", "y": r"$\dot{q}_2$ (m/s)"},
                            "CT": {"x": "Time (s)", "y": "Output"}}
# - Available options: configuration value; see description above.
PLOT_MODEL_COLORS = {"autograd": "tab:blue", "mgradnet_m": "tab:orange"}
# - Available options: string value; see description above and relevant configuration class.
PLOT_GROUND_TRUTH_COLOR = "black"
# - Available options: string value; see description above and relevant configuration class.
PLOT_GROUND_TRUTH_LINESTYLE = "--"
# - Available options: string value; see description above and relevant configuration class.
PLOT_TRAIN_LEGEND_LOCATION = "upper right"
# - Available options: string value; see description above and relevant configuration class.
PLOT_ROLLOUT_LEGEND_LOCATION = "upper right"
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
PLOT_TRAIN_LEGEND_FONTSIZE = 16
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
PLOT_ROLLOUT_LEGEND_FONTSIZE = 16
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
PLOT_AXIS_LABEL_FONTSIZE = 16
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
PLOT_TICK_LABEL_FONTSIZE = 16
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
PLOT_TITLE_FONTSIZE = 16
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
PLOT_FIGURE_DPI = 180
# - Available options: None (disabled/automatic), or a compatible value described above.
PLOT_TRAIN_XLIMITS = None
# - Available options: None (disabled/automatic), or a compatible value described above.
PLOT_TRAIN_YLIMITS = None
# - Available options: configuration value; see description above.
PLOT_ROLLOUT_YLIMITS = {"MSD": None, "CT": None}
# - Available options: tuple of values with the meaning and units described above.
PLOT_CT_TIME_LIMITS = (0, 4100)
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
PLOT_MSD_ROLLOUT_CHUNK_SIZE = 1000
# - Available options: True (enable) or False (disable).
PLOT_MSD_SHOW_WARMUP = True
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
PLOT_MSD_POST_WARMUP_START_EPOCH = 6
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
PLOT_MSD_WARMUP_END_EPOCH = 5
# - Available options: True (enable) or False (disable).
PLOT_MSD_MARK_WARMUP = False


# ==================== 3. MSD: SYSTEM CONFIGURATION ====================
# Only the MSD_* values in this section are editable.

# --- MSD / DATA GENERATION AND LOADING ---
# Time-series source, physics, simulation, splitting, noise and normalization.
# Directory containing source data files; MSD may generate data, CT reads measured records.
# - Available options: Path(...); RECONSTRUCTION_DIRECTORY also accepts None.
MSD_DATA_DATA_ROOT = Path('data/autograd_vs_gradnet_datasets')
# Dataset preparation mode: continuous simulation or measured record.
# - Available options: "continuous" (simulation), "measured" (existing records; system-dependent).
MSD_DATA_GENERATION_MODE = 'continuous'
# Random seed for dataset generation, independent of neural initialization.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_DATA_DATA_SEED = 123
# Seed for deterministic train/validation/test partitioning.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_DATA_SPLIT_SEED = 456
# Total MSD trajectories generated before selecting the active training subset.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_DATA_TRAIN_TRAJECTORIES = 200
# Number of MSD training trajectories actually used.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_DATA_ACTIVE_TRAIN_TRAJECTORIES = 20
# Number of validation trajectories.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_DATA_VALIDATION_TRAJECTORIES = 8
# Number of held-out test trajectories.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_DATA_TEST_TRAJECTORIES = 20
# Number of time samples in each simulated trajectory.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_DATA_TRAJECTORY_LENGTH = 1000
# Sampling interval of observed data in seconds.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_DATA_SAMPLING_TIME = 0.1
# Integrator for generating synthetic physical trajectories.
# - Available options: string value; see description above and relevant configuration class.
MSD_DATA_SIMULATION_INTEGRATOR = 'rk4'
# Number of numerical integration substeps per sample during simulation.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_DATA_SIMULATION_SUBSTEPS = 1
# Floating-point precision for physical simulation.
# - Available options: "float32", "float64" (subject to model/device support).
MSD_DATA_SIMULATION_DTYPE = 'float32'
# Compute device used for physical simulation.
# - Available options: "cpu" or a supported PyTorch device such as "cuda" or "mps".
MSD_DATA_SIMULATION_DEVICE = 'cpu'
# Uniform initial position interval for synthetic MSD.
# - Available options: tuple of values with the meaning and units described above.
MSD_DATA_INITIAL_POSITION_RANGE = (-1.0, 1.0)
# Uniform initial velocity interval for synthetic MSD.
# - Available options: tuple of values with the meaning and units described above.
MSD_DATA_INITIAL_VELOCITY_RANGE = (-0.5, 0.5)
# Base frequency for multisine control excitation.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_DATA_CONTROL_BASE_FREQUENCY = 0.01
# Number of harmonics in synthetic multisine excitation.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_DATA_CONTROL_NUM_HARMONICS = 100
# Amplitude scale for synthetic excitation.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_DATA_CONTROL_AMPLITUDE = 0.5
# Probability of zero-input intervals in synthetic control.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_DATA_CONTROL_ZERO_PROBABILITY = 0.0
# Candidate durations of zero-control intervals.
# - Available options: tuple of values with the meaning and units described above.
MSD_DATA_CONTROL_ZERO_INTERVAL_DURATIONS = (0.1, 0.15, 0.25, 0.5)
# Samples per synthetic control phase segment.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_DATA_CONTROL_PHASE_SEGMENT_LENGTH = 1000
# Number of segments in the continuous MSD training record.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_DATA_CONTINUOUS_TRAIN_SEGMENTS = 20
# Number of segments in the continuous MSD validation record.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_DATA_CONTINUOUS_VALIDATION_SEGMENTS = 8
# Number of segments in the continuous MSD test record.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_DATA_CONTINUOUS_TEST_SEGMENTS = 20
# Whether to scale synthetic control input according to the generator.
# - Available options: True (enable) or False (disable).
MSD_DATA_CONTROL_SCALE_INPUT = True
# Physical MSD coefficients passed directly to the simulator.
# - Available options: dictionary of physical system coefficients.
MSD_DATA_PHYSICAL_PARAMETERS = {
            'm1': 1.0,
            'm2': 1.0,
            'L1': 0.0,
            'L2': 0.0,
            'd_w': 0.0,
            'd_12': 0.0,
            'k_01': 1.0,
            'k_02': 1.0,
            'k_c1': 0.1,
            'k_c2': 0.0,
            'c_01': 0.5,
            'c_02': 0.5,
            'c_n1': 0.0,
            'c_n2': 0.0,
            'K_w': 0.0,
            'chi_w': 0.0,
            'K_12': 0.0,
            'chi_12': 0.0,
            'contact_exponent': 1.5,
        }
# Timing convention of the source input/output records.
# - Available options: string value; see description above and relevant configuration class.
MSD_DATA_SOURCE_ALIGNMENT = 'simultaneous'
# Whether output at a step is computed before state integration.
# - Available options: True (enable) or False (disable).
MSD_DATA_OUTPUT_BEFORE_INTEGRATION = True
# Training-output synthetic noise SNR in dB (None disables where supported).
# - Available options: SNR in dB (float), or None for noise-free output.
# MSD / DATA PARAMETRISATION — noise injected into generated outputs.
MSD_DATA_NOISE_SNR_TRAIN = 50.0
# Validation-output synthetic noise SNR in dB (None disables where supported).
# - Available options: SNR in dB (float), or None for noise-free output.
MSD_DATA_NOISE_SNR_VALIDATION = 50.0
# Test-output synthetic noise SNR in dB (None disables where supported).
# - Available options: SNR in dB (float), or None for noise-free output.
MSD_DATA_NOISE_SNR_TEST = None
# Random seed for synthetic training-output noise.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_DATA_NOISE_SEED_TRAIN = 1000
# Random seed for synthetic validation-output noise.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_DATA_NOISE_SEED_VALIDATION = 1001
# Random seed for synthetic test-output noise.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_DATA_NOISE_SEED_TEST = 1002
# Normalization settings (all selected methods share these).
# Master normalization switch: False disables input/output normalization for all selected methods.
# - Available options: True (enable) or False (disable).
# MSD / DATA PARAMETRISATION — normalization and metric units.
MSD_DATA_NORMALIZATION_ENABLED = False
# Normalize input only when normalization.enabled is True.
# - Available options: True (enable) or False (disable).
MSD_DATA_NORMALIZATION_NORMALIZE_INPUT = True
# Normalize output only when normalization.enabled is True.
# - Available options: True (enable) or False (disable).
MSD_DATA_NORMALIZATION_NORMALIZE_OUTPUT = True
# Source for normalization mean/scale (reference or training).
# - Available options: "reference" or another source supported by NormalizationConfig.
MSD_DATA_NORMALIZATION_STATISTICS_SOURCE = 'reference'
# Small positive floor used to avoid division by zero in normalization.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_DATA_NORMALIZATION_EPSILON = 1e-08
# Whether to inverse-transform predictions when normalization is active.
# - Available options: True (enable) or False (disable).
MSD_DATA_NORMALIZATION_INVERSE_TRANSFORM_PREDICTIONS = True
# Whether to inverse-transform targets when normalization is active.
# - Available options: True (enable) or False (disable).
MSD_DATA_NORMALIZATION_INVERSE_TRANSFORM_TARGETS = True
# Whether evaluation metrics are computed in physical units.
# - Available options: True (enable) or False (disable).
MSD_DATA_NORMALIZATION_PHYSICAL_UNIT_METRICS = True
# Reference record used for normalized-RMSE denominator.
# - Available options: "full_record" or a denominator supported by evaluation configuration.
MSD_DATA_NORMALIZATION_NRMSE_DENOMINATOR = 'full_record'

# --- MSD / MODEL PARAMETRISATION ---
# Architecture, Hamiltonian, port-Hamiltonian blocks and numerical integration.
# Dimension of the learned latent state.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_MODEL_STATE_DIM = 4
# Number of input channels.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_MODEL_INPUT_DIM = 1
# Number of observed output channels.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_MODEL_OUTPUT_DIM = 1
# Model floating-point precision.
# - Available options: "float32", "float64" (subject to model/device support).
MSD_MODEL_DTYPE = 'float32'
# Model compute device.
# - Available options: "cpu" or a supported PyTorch device such as "cuda" or "mps".
MSD_MODEL_DEVICE = 'cpu'
# Initial-state encoder architecture (MLP or convolutional).
# - Available options: "mlp", "cnn" (choose history encoder architecture).
MSD_MODEL_ENCODER_TYPE = 'mlp'
# MLP encoder hidden widths; ignored for convolutional encoder.
# - Available options: tuple of positive integers; layer widths, channels, kernels or pool sizes.
MSD_MODEL_ENCODER_HIDDEN_LAYERS = (64, 64)
# Convolutional encoder channel widths; ignored for MLP encoder.
# - Available options: tuple of positive integers; layer widths, channels, kernels or pool sizes.
MSD_MODEL_ENCODER_CONV_CHANNELS = (16, 32, 64)
# Convolution kernel lengths in samples.
# - Available options: tuple of positive integers; layer widths, channels, kernels or pool sizes.
MSD_MODEL_ENCODER_KERNEL_SIZES = (37, 5, 5)
# Convolution pooling factors.
# - Available options: tuple of positive integers; layer widths, channels, kernels or pool sizes.
MSD_MODEL_ENCODER_POOL_SIZES = (2, 2, 1)
# Nonlinearity in the encoder.
# - Available options: "tanh", "relu" (encoder hidden-layer nonlinearity).
MSD_MODEL_ENCODER_ACTIVATION = 'tanh'
# Dropout probability in the encoder.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_MODEL_ENCODER_DROPOUT = 0.0
# Hidden layer widths of the learned J block.
# - Available options: tuple of positive integers; layer widths, channels, kernels or pool sizes.
MSD_MODEL_INTERCONNECTION_HIDDEN_LAYERS = (8,)
# Hidden layer widths of the learned R block.
# - Available options: tuple of positive integers; layer widths, channels, kernels or pool sizes.
MSD_MODEL_DISSIPATION_HIDDEN_LAYERS = (8,)
# Hidden layer widths of the learned G block.
# - Available options: tuple of positive integers; layer widths, channels, kernels or pool sizes.
MSD_MODEL_PORT_HIDDEN_LAYERS = (8,)
# Hidden layer widths of the Hamiltonian network.
# - Available options: tuple of positive integers; layer widths, channels, kernels or pool sizes.
MSD_MODEL_HAMILTONIAN_HIDDEN_LAYERS = (16, 16)
# Nonlinearity used by the PH neural components.
# - Available options: "tanh", "relu" (shared MLP nonlinearity).
MSD_MODEL_ACTIVATION = 'tanh'
# Lower-bound parameter for Hamiltonian representation.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_MODEL_HAMILTONIAN_LOWER_BOUND = 0.0
# Multiplier applied to final-layer initialization.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_MODEL_FINAL_LAYER_INITIALIZATION_SCALE = 1.0
# Hidden dimension of GradNet Hamiltonian modules.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_MODEL_GRADNET_HIDDEN_DIM = 4
# Number of GradNet Hamiltonian modules.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_MODEL_GRADNET_NUM_MODULES = 4
# Numerical floor for GradNet operations.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_MODEL_GRADNET_FLOOR_EPS = 1e-06
# Per-method multiplicative scaling of Hamiltonian gradients.
# - Available options: dictionary mapping named components or methods to their settings.
MSD_MODEL_GRADIENT_SCALES = {
            'autograd': 1.0,
            'mgradnet_m': 1.0,
            'simple_mgradnet_m': 1.0,
            'gradnet_m': 1.0,
        }
# Integrator used in learned model rollouts.
# - Available options: "rk4" (integration algorithm).
MSD_MODEL_INTEGRATOR = 'rk4'
# Numerical integration substeps per model sample.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_MODEL_INTEGRATOR_SUBSTEPS = 1
# Model integration sampling time in seconds; can differ from observed data.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_MODEL_MODEL_SAMPLING_TIME = 0.1
# Method for obtaining rollout initial state (typically encoder).
# - Available options: "encoder" or another mode accepted by ModelConfig.
MSD_MODEL_INITIAL_STATE_MODE = 'encoder'
# Whether PH components are neural or fixed.
# - Available options: dictionary mapping named components or methods to their settings.
MSD_MODEL_COMPONENT_MODES = {
            'hamiltonian': 'neural',
            'interconnection': 'neural',
            'dissipation': 'neural',
            'port': 'neural',
        }
# Names of model components whose parameters are optimized.
# - Available options: tuple drawn from encoder, interconnection, dissipation, port, hamiltonian.
MSD_MODEL_TRAINABLE_COMPONENTS = ('encoder', 'interconnection', 'dissipation', 'port', 'hamiltonian')
# Factorization rank used for learned dissipation.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_MODEL_DISSIPATION_FACTOR_RANK = 2
# Legacy stored offsets. Active initialization uses SEED_MODE and MANUAL_COMPONENT_SEEDS, not these offsets.
# - Available options: dictionary mapping named components or methods to their settings.
MSD_MODEL_COMPONENT_SEED_OFFSETS = {
            'encoder': 18,
            'interconnection': 12,
            'dissipation': 11,
            'port': 13,
            'autograd_hamiltonian': 14,
            'gradnet_hamiltonian': 15,
        }

# --- MSD / DATA PARAMETRISATION: RUNTIME WINDOWING ---
# History and rollout lengths, strides and window selection.
# Number of past input/output samples provided to the encoder.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_WINDOWS_HISTORY_LENGTH = 20
# Final fixed-horizon rollout length in samples. For curriculum, final length is derived from the last schedule entry.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_WINDOWS_PREDICTION_HORIZON = 200
# Sample-step spacing between successive window start indices.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_WINDOWS_STRIDE = 1
# Fraction of candidate training windows retained.
# - Available options: float in (0,1] selecting the available windows.
MSD_WINDOWS_TRAIN_FRACTION = 1.0
# Fraction of candidate validation windows retained.
# - Available options: float in (0,1] selecting the available windows.
MSD_WINDOWS_VALIDATION_FRACTION = 1.0
# Fraction of candidate test windows retained.
# - Available options: float in (0,1] selecting the available windows.
MSD_WINDOWS_TEST_FRACTION = 1.0
# Seed for deterministic window subsampling.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_WINDOWS_SELECTION_SEED = 42

# --- MSD / TRAIN-VALIDATION-TEST PARAMETRISATION ---
# Epoch phases, minibatches, validation, curricula and phase-boundary actions.
# Number of training windows per optimizer minibatch.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_TRAINING_BATCH_SIZE = 256
# Whether to shuffle training windows each epoch.
# - Available options: True (enable) or False (disable).
MSD_TRAINING_SHUFFLE = True
# Initial seed for window shuffling.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_TRAINING_SHUFFLE_SEED = 42
# Rule controlling shuffle seed changes.
# - Available options: "fixed" or another policy supported by TrainingConfig.
MSD_TRAINING_SHUFFLE_SEED_POLICY = 'fixed'
# Number of data-loading worker processes.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_TRAINING_NUM_WORKERS = 0
# Maximum gradient norm; None disables clipping.
# - Available options: None (disabled/automatic), or a compatible value described above.
MSD_TRAINING_GRADIENT_CLIP_NORM = None
# Number of epochs without validation improvement before stopping; None disables.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_TRAINING_EARLY_STOPPING_PATIENCE = 70
# Validation on windows, full rollout, or weighted mixture.
# - Available options: "windowed", "full_rollout", "mixed".
MSD_TRAINING_VALIDATION_MODE = 'full_rollout'
# Weight of full-rollout validation in mixed mode.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_TRAINING_VALIDATION_FULL_ROLLOUT_WEIGHT = 0.1  # recorded; full_rollout mode uses full rollout
# Reference record used for validation NRMSE denominator.
# - Available options: string value; see description above and relevant configuration class.
MSD_TRAINING_VALIDATION_NRMSE_DENOMINATOR = 'full_record'
# Whether validation metrics are computed in physical units.
# - Available options: True (enable) or False (disable).
MSD_TRAINING_VALIDATION_PHYSICAL_METRICS = False
# Seconds skipped at beginning of validation evaluation.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_TRAINING_VALIDATION_SKIP_INITIAL_TIME = 0.0
# Seconds skipped at beginning of test evaluation.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_TRAINING_TEST_SKIP_INITIAL_TIME = 0.0
# Test segment duration in seconds; None uses full segment.
# - Available options: None (disabled/automatic), or a compatible value described above.
MSD_TRAINING_TEST_SEGMENT_LENGTH_TIME = 10.0  # MSD PH ablation
# Whether to discard incomplete final minibatches.
# - Available options: True (enable) or False (disable).
MSD_TRAINING_DROP_LAST = False
# Consecutive phase durations in epochs. Their sum is the total number of epochs.
# - Available options: tuple of positive integer epoch-block lengths.
MSD_TRAINING_FIT_PHASES = (10, 10, 10, 10, 10, 20)
# None: train at fixed horizon. Otherwise {zero-based epoch: active horizon}; final schedule value defines the final horizon. For horizon ablation, fixed horizons are used.
# - Available options: None where supported, or {epoch_index: value} schedule.
MSD_TRAINING_PREDICTION_HORIZON_SCHEDULE = {0: 5}#, 4: 200}  # original MSD warm-up: one-based epochs 1 and 5
# Epoch-based limits on state derivative; None disables.
# - Available options: None (disabled/automatic), or a compatible value described above.
MSD_TRAINING_STATE_DERIVATIVE_LIMIT_SCHEDULE = {0: 10000.0}  # MSD PH ablation: one-based epoch 1
# Epoch (zero-based) -> learning rate. An entry at epoch 0 takes precedence over initial learning_rate.
# Master seed for phase-specific shuffling; None uses default policy.
# - Available options: None (disabled/automatic), or a compatible value described above.
MSD_TRAINING_PHASE_SHUFFLE_MASTER_SEED = None
# Whether to restore training history across phases.
# - Available options: True (enable) or False (disable).
MSD_TRAINING_RESTORE_HISTORY_BETWEEN_PHASES = False
# Restore best validation model at phase boundaries.
# - Available options: True (enable) or False (disable).
MSD_TRAINING_RELOAD_BEST_AT_PHASE_BOUNDARY = False
# Reshuffle training windows at each phase boundary.
# - Available options: True (enable) or False (disable).
MSD_TRAINING_RESHUFFLE_AT_PHASE_BOUNDARY = False
# Whether to evaluate validation loss before the first epoch.
# - Available options: True (enable) or False (disable).
MSD_TRAINING_VALIDATE_INITIAL = False
# Whether to square model outputs before loss computation.
# - Available options: True (enable) or False (disable).
MSD_TRAINING_SQUARE_OUTPUTS = False

# --- MSD / TRAIN-VALIDATION-TEST: OPTIMIZERS ---
# Optimizer and learning-rate policy for each method.

# autograd: optimizer settings (only selected METHODS are used).
# Optimizer algorithm name (Adam in the supplied configuration).
# - Available options: "Adam" or an optimizer name supported by the training implementation.
MSD_OPT_AUTOGRAD_NAME = 'Adam'
# Initial optimizer learning rate; if an LR schedule contains epoch 0, its value is the effective initial LR.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_OPT_AUTOGRAD_LEARNING_RATE = 0.001
# Optimizer weight decay coefficient.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_OPT_AUTOGRAD_WEIGHT_DECAY = 0.0
# Adam first- and second-moment decay factors.
# - Available options: pair of floats in [0,1) (Adam momentum coefficients).
MSD_OPT_AUTOGRAD_BETAS = (0.9, 0.999)
# Adam numerical stability constant.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_OPT_AUTOGRAD_EPS = 1e-08
# Enable AMSGrad variant of Adam.
# - Available options: True (enable) or False (disable).
MSD_OPT_AUTOGRAD_AMSGRAD = False
# Epoch (zero-based) -> learning rate. An entry at epoch 0 takes precedence over initial learning_rate.
# - Available options: None where supported, or {epoch_index: value} schedule.
MSD_OPT_AUTOGRAD_LEARNING_RATE_SCHEDULE = {
                0: 0.001,
            }
# Optional optimizer scheduler type; None disables.
# - Available options: None (no scheduler), or a scheduler supported by the optimizer configuration.
MSD_OPT_AUTOGRAD_SCHEDULER_TYPE = None
# Keyword parameters passed to optional scheduler.
# - Available options: dictionary of scheduler keyword parameters; {} when disabled.
MSD_OPT_AUTOGRAD_SCHEDULER_PARAMETERS = {}

# mgradnet_m: optimizer settings (only selected METHODS are used).
# Optimizer algorithm name (Adam in the supplied configuration).
# - Available options: "Adam" or an optimizer name supported by the training implementation.
MSD_OPT_MGRADNET_M_NAME = 'Adam'
# Initial optimizer learning rate; if an LR schedule contains epoch 0, its value is the effective initial LR.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_OPT_MGRADNET_M_LEARNING_RATE = 0.01
# Optimizer weight decay coefficient.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_OPT_MGRADNET_M_WEIGHT_DECAY = 0.0
# Adam first- and second-moment decay factors.
# - Available options: pair of floats in [0,1) (Adam momentum coefficients).
MSD_OPT_MGRADNET_M_BETAS = (0.9, 0.999)
# Adam numerical stability constant.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
MSD_OPT_MGRADNET_M_EPS = 1e-08
# Enable AMSGrad variant of Adam.
# - Available options: True (enable) or False (disable).
MSD_OPT_MGRADNET_M_AMSGRAD = False
# Epoch (zero-based) -> learning rate. An entry at epoch 0 takes precedence over initial learning_rate.
# - Available options: None where supported, or {epoch_index: value} schedule.
MSD_OPT_MGRADNET_M_LEARNING_RATE_SCHEDULE = {
                0: 0.01,
                20: 0.005,
                30: 0.002,
                40: 0.001,
            }
# Optional optimizer scheduler type; None disables.
# - Available options: None (no scheduler), or a scheduler supported by the optimizer configuration.
MSD_OPT_MGRADNET_M_SCHEDULER_TYPE = None
# Keyword parameters passed to optional scheduler.
# - Available options: dictionary of scheduler keyword parameters; {} when disabled.
MSD_OPT_MGRADNET_M_SCHEDULER_PARAMETERS = {}

# ==================== 4. CT: SYSTEM CONFIGURATION ====================
# Only the CT_* values in this section are editable.

# --- CT / DATA LOADING ---
# Time-series source, physics, simulation, splitting, noise and normalization.
# Directory containing source data files; MSD may generate data, CT reads measured records.
# - Available options: Path(...); RECONSTRUCTION_DIRECTORY also accepts None.
CT_DATA_DATA_ROOT = Path('data/TANKS')
# Filename of measured dataset (CT); None for generated MSD.
# - Available options: string value; see description above and relevant configuration class.
CT_DATA_DATASET_FILENAME = 'dataBenchmark.csv'
# Dataset preparation mode: continuous simulation or measured record.
# - Available options: "continuous" (simulation), "measured" (existing records; system-dependent).
CT_DATA_GENERATION_MODE = 'measured'
# Sampling interval of observed data in seconds.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
CT_DATA_SAMPLING_TIME = 4.0
# Names of input/output columns for measured training data.
# - Available options: tuple of values with the meaning and units described above.
CT_DATA_TRAIN_COLUMNS = ('uEst', 'yEst')
# Names of input/output columns for measured validation data.
# - Available options: tuple of values with the meaning and units described above.
CT_DATA_VALIDATION_COLUMNS = ('uVal', 'yVal')
# Name of the measured sampling-time column.
# - Available options: string value; see description above and relevant configuration class.
CT_DATA_SAMPLING_TIME_COLUMN = 'Ts'
# Number of raw CT validation samples; None uses configured default.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
CT_DATA_VALIDATION_RAW_SAMPLES = 512
# Normalization settings (all selected methods share these).
# Master normalization switch: False disables input/output normalization for all selected methods.
# - Available options: True (enable) or False (disable).
# CT / DATA PARAMETRISATION — normalization and metric units.
CT_DATA_NORMALIZATION_ENABLED = True
# Normalize input only when normalization.enabled is True.
# - Available options: True (enable) or False (disable).
CT_DATA_NORMALIZATION_NORMALIZE_INPUT = True
# Normalize output only when normalization.enabled is True.
# - Available options: True (enable) or False (disable).
CT_DATA_NORMALIZATION_NORMALIZE_OUTPUT = True
# Source for normalization mean/scale (reference or training).
# - Available options: "reference" or another source supported by NormalizationConfig.
CT_DATA_NORMALIZATION_STATISTICS_SOURCE = 'reference'
# Small positive floor used to avoid division by zero in normalization.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
CT_DATA_NORMALIZATION_EPSILON = 1e-08
# Whether to inverse-transform predictions when normalization is active.
# - Available options: True (enable) or False (disable).
CT_DATA_NORMALIZATION_INVERSE_TRANSFORM_PREDICTIONS = True
# Whether to inverse-transform targets when normalization is active.
# - Available options: True (enable) or False (disable).
CT_DATA_NORMALIZATION_INVERSE_TRANSFORM_TARGETS = True
# Whether evaluation metrics are computed in physical units.
# - Available options: True (enable) or False (disable).
CT_DATA_NORMALIZATION_PHYSICAL_UNIT_METRICS = True
# Reference record used for normalized-RMSE denominator.
# - Available options: "full_record" or a denominator supported by evaluation configuration.
CT_DATA_NORMALIZATION_NRMSE_DENOMINATOR = 'full_record'

# --- CT / MODEL PARAMETRISATION ---
# Architecture, Hamiltonian, port-Hamiltonian blocks and numerical integration.
# Dimension of the learned latent state.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
CT_MODEL_STATE_DIM = 2
# Number of input channels.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
CT_MODEL_INPUT_DIM = 1
# Number of observed output channels.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
CT_MODEL_OUTPUT_DIM = 1
# Model floating-point precision.
# - Available options: "float32", "float64" (subject to model/device support).
CT_MODEL_DTYPE = 'float32'
# Model compute device.
# - Available options: "cpu" or a supported PyTorch device such as "cuda" or "mps".
CT_MODEL_DEVICE = 'cpu'
# Initial-state encoder architecture (MLP or convolutional).
# - Available options: "mlp", "cnn" (choose history encoder architecture).
CT_MODEL_ENCODER_TYPE = 'mlp'
# MLP encoder hidden widths; ignored for convolutional encoder.
# - Available options: tuple of positive integers; layer widths, channels, kernels or pool sizes.
CT_MODEL_ENCODER_HIDDEN_LAYERS = (64, 64)
# Convolutional encoder channel widths; ignored for MLP encoder.
# - Available options: tuple of positive integers; layer widths, channels, kernels or pool sizes.
CT_MODEL_ENCODER_CONV_CHANNELS = (8,)
# Convolution kernel lengths in samples.
# - Available options: tuple of positive integers; layer widths, channels, kernels or pool sizes.
CT_MODEL_ENCODER_KERNEL_SIZES = (3,)
# Convolution pooling factors.
# - Available options: tuple of positive integers; layer widths, channels, kernels or pool sizes.
CT_MODEL_ENCODER_POOL_SIZES = (1,)
# Nonlinearity in the encoder.
# - Available options: "tanh", "relu" (encoder hidden-layer nonlinearity).
CT_MODEL_ENCODER_ACTIVATION = 'tanh'
# Dropout probability in the encoder.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
CT_MODEL_ENCODER_DROPOUT = 0.0
# Hidden layer widths of the learned J block.
# - Available options: tuple of positive integers; layer widths, channels, kernels or pool sizes.
CT_MODEL_INTERCONNECTION_HIDDEN_LAYERS = (8,)
# Hidden layer widths of the learned R block.
# - Available options: tuple of positive integers; layer widths, channels, kernels or pool sizes.
CT_MODEL_DISSIPATION_HIDDEN_LAYERS = (8,)
# Hidden layer widths of the learned G block.
# - Available options: tuple of positive integers; layer widths, channels, kernels or pool sizes.
CT_MODEL_PORT_HIDDEN_LAYERS = (8,)
# Hidden layer widths of the Hamiltonian network.
# - Available options: tuple of positive integers; layer widths, channels, kernels or pool sizes.
CT_MODEL_HAMILTONIAN_HIDDEN_LAYERS = (8,)
# Nonlinearity used by the PH neural components.
# - Available options: "tanh", "relu" (shared MLP nonlinearity).
CT_MODEL_ACTIVATION = 'tanh'
# Lower-bound parameter for Hamiltonian representation.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
CT_MODEL_HAMILTONIAN_LOWER_BOUND = 0.0
# Multiplier applied to final-layer initialization.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
CT_MODEL_FINAL_LAYER_INITIALIZATION_SCALE = 1.0
# Hidden dimension of GradNet Hamiltonian modules.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
CT_MODEL_GRADNET_HIDDEN_DIM = 2
# Number of GradNet Hamiltonian modules.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
CT_MODEL_GRADNET_NUM_MODULES = 2
# Numerical floor for GradNet operations.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
CT_MODEL_GRADNET_FLOOR_EPS = 1e-06
# Per-method multiplicative scaling of Hamiltonian gradients.
# - Available options: dictionary mapping named components or methods to their settings.
CT_MODEL_GRADIENT_SCALES = {
            'autograd': 1.0,
            'mgradnet_m': 1.0,
            'simple_mgradnet_m': 1.0,
            'gradnet_m': 1.0,
        }
# Integrator used in learned model rollouts.
# - Available options: "rk4" (integration algorithm).
CT_MODEL_INTEGRATOR = 'rk4'
# Numerical integration substeps per model sample.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
CT_MODEL_INTEGRATOR_SUBSTEPS = 1
# Model integration sampling time in seconds; can differ from observed data.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
CT_MODEL_MODEL_SAMPLING_TIME = 0.04
# Method for obtaining rollout initial state (typically encoder).
# - Available options: "encoder" or another mode accepted by ModelConfig.
CT_MODEL_INITIAL_STATE_MODE = 'encoder'
# Whether PH components are neural or fixed.
# - Available options: dictionary mapping named components or methods to their settings.
CT_MODEL_COMPONENT_MODES = {
            'hamiltonian': 'neural',
            'interconnection': 'neural',
            'dissipation': 'neural',
            'port': 'neural',
        }
# Names of model components whose parameters are optimized.
# - Available options: tuple drawn from encoder, interconnection, dissipation, port, hamiltonian.
CT_MODEL_TRAINABLE_COMPONENTS = ('encoder', 'interconnection', 'dissipation', 'port', 'hamiltonian')
# Factorization rank used for learned dissipation.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
CT_MODEL_DISSIPATION_FACTOR_RANK = 2
# Legacy stored offsets. Active initialization uses SEED_MODE and MANUAL_COMPONENT_SEEDS, not these offsets.
# - Available options: dictionary mapping named components or methods to their settings.
CT_MODEL_COMPONENT_SEED_OFFSETS = {
            'encoder': 18,
            'interconnection': 12,
            'dissipation': 11,
            'port': 13,
            'autograd_hamiltonian': 14,
            'gradnet_hamiltonian': 15,
        }

# --- CT / DATA PARAMETRISATION: RUNTIME WINDOWING ---
# History and rollout lengths, strides and window selection.
# Number of past input/output samples provided to the encoder.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
CT_WINDOWS_HISTORY_LENGTH = 4
# Final fixed-horizon rollout length in samples. For curriculum, final length is derived from the last schedule entry.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
CT_WINDOWS_PREDICTION_HORIZON = 60
# Sample-step spacing between successive window start indices.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
CT_WINDOWS_STRIDE = 1
# Fraction of candidate training windows retained.
# - Available options: float in (0,1] selecting the available windows.
CT_WINDOWS_TRAIN_FRACTION = 1.0
# Fraction of candidate validation windows retained.
# - Available options: float in (0,1] selecting the available windows.
CT_WINDOWS_VALIDATION_FRACTION = 1.0
# Fraction of candidate test windows retained.
# - Available options: float in (0,1] selecting the available windows.
CT_WINDOWS_TEST_FRACTION = 1.0
# Seed for deterministic window subsampling.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
CT_WINDOWS_SELECTION_SEED = 0

# --- CT / TRAIN-VALIDATION-TEST PARAMETRISATION ---
# Epoch phases, minibatches, validation, curricula and phase-boundary actions.
# Number of training windows per optimizer minibatch.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
CT_TRAINING_BATCH_SIZE = 32
# Whether to shuffle training windows each epoch.
# - Available options: True (enable) or False (disable).
CT_TRAINING_SHUFFLE = True
# Initial seed for window shuffling.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
CT_TRAINING_SHUFFLE_SEED = 0
# Rule controlling shuffle seed changes.
# - Available options: "fixed" or another policy supported by TrainingConfig.
CT_TRAINING_SHUFFLE_SEED_POLICY = 'fixed'
# Number of data-loading worker processes.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
CT_TRAINING_NUM_WORKERS = 0
# Maximum gradient norm; None disables clipping.
# - Available options: None (disabled/automatic), or a compatible value described above.
CT_TRAINING_GRADIENT_CLIP_NORM = None
# Number of epochs without validation improvement before stopping; None disables.
# - Available options: None (disabled/automatic), or a compatible value described above.
CT_TRAINING_EARLY_STOPPING_PATIENCE = 700
# Validation on windows, full rollout, or weighted mixture.
# - Available options: "windowed", "full_rollout", "mixed".
CT_TRAINING_VALIDATION_MODE = 'mixed'
# Weight of full-rollout validation in mixed mode.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
CT_TRAINING_VALIDATION_FULL_ROLLOUT_WEIGHT = 0.1
# Reference record used for validation NRMSE denominator.
# - Available options: string value; see description above and relevant configuration class.
CT_TRAINING_VALIDATION_NRMSE_DENOMINATOR = 'full_record'
# Whether validation metrics are computed in physical units.
# - Available options: True (enable) or False (disable).
CT_TRAINING_VALIDATION_PHYSICAL_METRICS = False
# Seconds skipped at beginning of validation evaluation.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
CT_TRAINING_VALIDATION_SKIP_INITIAL_TIME = 0.0
# Seconds skipped at beginning of test evaluation.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
CT_TRAINING_TEST_SKIP_INITIAL_TIME = 0.0
# Test segment duration in seconds; None uses full segment.
# - Available options: None (disabled/automatic), or a compatible value described above.
CT_TRAINING_TEST_SEGMENT_LENGTH_TIME = None
# Whether to discard incomplete final minibatches.
# - Available options: True (enable) or False (disable).
CT_TRAINING_DROP_LAST = True
# Consecutive phase durations in epochs. Their sum is the total number of epochs.
# - Available options: tuple of positive integer epoch-block lengths.
CT_TRAINING_FIT_PHASES = (10, 10, 10, 20, 20, 30, 50, 50, 50, 50, 100, 100, 200)
# None: train at fixed horizon. Otherwise {zero-based epoch: active horizon}; final schedule value defines the final horizon. For horizon ablation, fixed horizons are used.
# - Available options: None where supported, or {epoch_index: value} schedule.
CT_TRAINING_PREDICTION_HORIZON_SCHEDULE = None
# Epoch-based limits on state derivative; None disables.
# - Available options: None (disabled/automatic), or a compatible value described above.
CT_TRAINING_STATE_DERIVATIVE_LIMIT_SCHEDULE = None
# Master seed for phase-specific shuffling; None uses default policy.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
CT_TRAINING_PHASE_SHUFFLE_MASTER_SEED = 33
# Whether to restore training history across phases.
# - Available options: True (enable) or False (disable).
CT_TRAINING_RESTORE_HISTORY_BETWEEN_PHASES = True
# Restore best validation model at phase boundaries.
# - Available options: True (enable) or False (disable).
CT_TRAINING_RELOAD_BEST_AT_PHASE_BOUNDARY = True
# Reshuffle training windows at each phase boundary.
# - Available options: True (enable) or False (disable).
CT_TRAINING_RESHUFFLE_AT_PHASE_BOUNDARY = True
# Whether to evaluate validation loss before the first epoch.
# - Available options: True (enable) or False (disable).
CT_TRAINING_VALIDATE_INITIAL = True
# Whether to square model outputs before loss computation.
# - Available options: True (enable) or False (disable).
CT_TRAINING_SQUARE_OUTPUTS = False

# --- CT / TRAIN-VALIDATION-TEST: OPTIMIZERS ---
# Optimizer and learning-rate policy for each method.

# autograd: optimizer settings (only selected METHODS are used).
# Optimizer algorithm name (Adam in the supplied configuration).
# - Available options: "Adam" or an optimizer name supported by the training implementation.
CT_OPT_AUTOGRAD_NAME = 'Adam'
# Initial optimizer learning rate; if an LR schedule contains epoch 0, its value is the effective initial LR.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
CT_OPT_AUTOGRAD_LEARNING_RATE = 0.01
# Optimizer weight decay coefficient.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
CT_OPT_AUTOGRAD_WEIGHT_DECAY = 0.0
# Adam first- and second-moment decay factors.
# - Available options: pair of floats in [0,1) (Adam momentum coefficients).
CT_OPT_AUTOGRAD_BETAS = (0.9, 0.999)
# Adam numerical stability constant.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
CT_OPT_AUTOGRAD_EPS = 1e-08
# Enable AMSGrad variant of Adam.
# - Available options: True (enable) or False (disable).
CT_OPT_AUTOGRAD_AMSGRAD = False
# Epoch (zero-based) -> learning rate. An entry at epoch 0 takes precedence over initial learning_rate.
# - Available options: None where supported, or {epoch_index: value} schedule.
CT_OPT_AUTOGRAD_LEARNING_RATE_SCHEDULE = {0: 0.01, 10: 0.005, 30: 0.001, 600: 0.0002}  # CT PH ablation
# Optional optimizer scheduler type; None disables.
# - Available options: None (no scheduler), or a scheduler supported by the optimizer configuration.
CT_OPT_AUTOGRAD_SCHEDULER_TYPE = None
# Keyword parameters passed to optional scheduler.
# - Available options: dictionary of scheduler keyword parameters; {} when disabled.
CT_OPT_AUTOGRAD_SCHEDULER_PARAMETERS = {}

# mgradnet_m: optimizer settings (only selected METHODS are used).
# Optimizer algorithm name (Adam in the supplied configuration).
# - Available options: "Adam" or an optimizer name supported by the training implementation.
CT_OPT_MGRADNET_M_NAME = 'Adam'
# Initial optimizer learning rate; if an LR schedule contains epoch 0, its value is the effective initial LR.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
CT_OPT_MGRADNET_M_LEARNING_RATE = 0.01
# Optimizer weight decay coefficient.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
CT_OPT_MGRADNET_M_WEIGHT_DECAY = 0.0
# Adam first- and second-moment decay factors.
# - Available options: pair of floats in [0,1) (Adam momentum coefficients).
CT_OPT_MGRADNET_M_BETAS = (0.9, 0.999)
# Adam numerical stability constant.
# - Available options: numeric value; keep within the physical or algorithmic constraints described above.
CT_OPT_MGRADNET_M_EPS = 1e-08
# Enable AMSGrad variant of Adam.
# - Available options: True (enable) or False (disable).
CT_OPT_MGRADNET_M_AMSGRAD = False
# Epoch (zero-based) -> learning rate. An entry at epoch 0 takes precedence over initial learning_rate.
# - Available options: None where supported, or {epoch_index: value} schedule.
CT_OPT_MGRADNET_M_LEARNING_RATE_SCHEDULE = {
                0: 0.01,
                10: 0.005,
                30: 0.001,
                600: 0.0002,
            }
# Optional optimizer scheduler type; None disables.
# - Available options: None (no scheduler), or a scheduler supported by the optimizer configuration.
CT_OPT_MGRADNET_M_SCHEDULER_TYPE = None
# Keyword parameters passed to optional scheduler.
# - Available options: dictionary of scheduler keyword parameters; {} when disabled.
CT_OPT_MGRADNET_M_SCHEDULER_PARAMETERS = {}

# ===================== 5. FINAL CONFIGURATION ASSEMBLY (DO NOT EDIT) =====================
# GENERATED FROM THE MASTER PARAMETERS ABOVE. DO NOT EDIT BELOW THIS LINE.

# - Available options: configuration value; see description above.
MSD_CONFIG = {
    'data': {
        "system": 'MSD',
        'data_root': MSD_DATA_DATA_ROOT,
        'generation_mode': MSD_DATA_GENERATION_MODE,
        'data_seed': MSD_DATA_DATA_SEED,
        'split_seed': MSD_DATA_SPLIT_SEED,
        'train_trajectories': MSD_DATA_TRAIN_TRAJECTORIES,
        'active_train_trajectories': MSD_DATA_ACTIVE_TRAIN_TRAJECTORIES,
        'validation_trajectories': MSD_DATA_VALIDATION_TRAJECTORIES,
        'test_trajectories': MSD_DATA_TEST_TRAJECTORIES,
        'trajectory_length': MSD_DATA_TRAJECTORY_LENGTH,
        'sampling_time': MSD_DATA_SAMPLING_TIME,
        'simulation_integrator': MSD_DATA_SIMULATION_INTEGRATOR,
        'simulation_substeps': MSD_DATA_SIMULATION_SUBSTEPS,
        'simulation_dtype': MSD_DATA_SIMULATION_DTYPE,
        'simulation_device': MSD_DATA_SIMULATION_DEVICE,
        'initial_position_range': MSD_DATA_INITIAL_POSITION_RANGE,
        'initial_velocity_range': MSD_DATA_INITIAL_VELOCITY_RANGE,
        'control_base_frequency': MSD_DATA_CONTROL_BASE_FREQUENCY,
        'control_num_harmonics': MSD_DATA_CONTROL_NUM_HARMONICS,
        'control_amplitude': MSD_DATA_CONTROL_AMPLITUDE,
        'control_zero_probability': MSD_DATA_CONTROL_ZERO_PROBABILITY,
        'control_zero_interval_durations': MSD_DATA_CONTROL_ZERO_INTERVAL_DURATIONS,
        'control_phase_segment_length': MSD_DATA_CONTROL_PHASE_SEGMENT_LENGTH,
        'continuous_train_segments': MSD_DATA_CONTINUOUS_TRAIN_SEGMENTS,
        'continuous_validation_segments': MSD_DATA_CONTINUOUS_VALIDATION_SEGMENTS,
        'continuous_test_segments': MSD_DATA_CONTINUOUS_TEST_SEGMENTS,
        'control_scale_input': MSD_DATA_CONTROL_SCALE_INPUT,
        'physical_parameters': MSD_DATA_PHYSICAL_PARAMETERS,
        'source_alignment': MSD_DATA_SOURCE_ALIGNMENT,
        'output_before_integration': MSD_DATA_OUTPUT_BEFORE_INTEGRATION,
        'noise_snr_train': MSD_DATA_NOISE_SNR_TRAIN,
        'noise_snr_validation': MSD_DATA_NOISE_SNR_VALIDATION,
        'noise_snr_test': MSD_DATA_NOISE_SNR_TEST,
        'noise_seed_train': MSD_DATA_NOISE_SEED_TRAIN,
        'noise_seed_validation': MSD_DATA_NOISE_SEED_VALIDATION,
        'noise_seed_test': MSD_DATA_NOISE_SEED_TEST,
        'normalization': {
            'enabled': MSD_DATA_NORMALIZATION_ENABLED,
            'normalize_input': MSD_DATA_NORMALIZATION_NORMALIZE_INPUT,
            'normalize_output': MSD_DATA_NORMALIZATION_NORMALIZE_OUTPUT,
            'statistics_source': MSD_DATA_NORMALIZATION_STATISTICS_SOURCE,
            'epsilon': MSD_DATA_NORMALIZATION_EPSILON,
            'inverse_transform_predictions': MSD_DATA_NORMALIZATION_INVERSE_TRANSFORM_PREDICTIONS,
            'inverse_transform_targets': MSD_DATA_NORMALIZATION_INVERSE_TRANSFORM_TARGETS,
            'physical_unit_metrics': MSD_DATA_NORMALIZATION_PHYSICAL_UNIT_METRICS,
            'nrmse_denominator': MSD_DATA_NORMALIZATION_NRMSE_DENOMINATOR,
        },
    },
    'model': {
        'state_dim': MSD_MODEL_STATE_DIM,
        'input_dim': MSD_MODEL_INPUT_DIM,
        'output_dim': MSD_MODEL_OUTPUT_DIM,
        'dtype': MSD_MODEL_DTYPE,
        'device': MSD_MODEL_DEVICE,
        'encoder_type': MSD_MODEL_ENCODER_TYPE,
        'encoder_hidden_layers': MSD_MODEL_ENCODER_HIDDEN_LAYERS,
        'encoder_conv_channels': MSD_MODEL_ENCODER_CONV_CHANNELS,
        'encoder_kernel_sizes': MSD_MODEL_ENCODER_KERNEL_SIZES,
        'encoder_pool_sizes': MSD_MODEL_ENCODER_POOL_SIZES,
        'encoder_activation': MSD_MODEL_ENCODER_ACTIVATION,
        'encoder_dropout': MSD_MODEL_ENCODER_DROPOUT,
        'interconnection_hidden_layers': MSD_MODEL_INTERCONNECTION_HIDDEN_LAYERS,
        'dissipation_hidden_layers': MSD_MODEL_DISSIPATION_HIDDEN_LAYERS,
        'port_hidden_layers': MSD_MODEL_PORT_HIDDEN_LAYERS,
        'hamiltonian_hidden_layers': MSD_MODEL_HAMILTONIAN_HIDDEN_LAYERS,
        'activation': MSD_MODEL_ACTIVATION,
        'hamiltonian_lower_bound': MSD_MODEL_HAMILTONIAN_LOWER_BOUND,
        'final_layer_initialization_scale': MSD_MODEL_FINAL_LAYER_INITIALIZATION_SCALE,
        'gradnet_hidden_dim': MSD_MODEL_GRADNET_HIDDEN_DIM,
        'gradnet_num_modules': MSD_MODEL_GRADNET_NUM_MODULES,
        'gradnet_floor_eps': MSD_MODEL_GRADNET_FLOOR_EPS,
        'gradient_scales': MSD_MODEL_GRADIENT_SCALES,
        'integrator': MSD_MODEL_INTEGRATOR,
        'integrator_substeps': MSD_MODEL_INTEGRATOR_SUBSTEPS,
        'model_sampling_time': MSD_MODEL_MODEL_SAMPLING_TIME,
        'initial_state_mode': MSD_MODEL_INITIAL_STATE_MODE,
        'component_modes': MSD_MODEL_COMPONENT_MODES,
        'trainable_components': MSD_MODEL_TRAINABLE_COMPONENTS,
        'dissipation_factor_rank': MSD_MODEL_DISSIPATION_FACTOR_RANK,
        'component_seed_offsets': MSD_MODEL_COMPONENT_SEED_OFFSETS,
    },
    'windows': {
        'history_length': MSD_WINDOWS_HISTORY_LENGTH,
        'prediction_horizon': (list(MSD_TRAINING_PREDICTION_HORIZON_SCHEDULE.values())[-1] if MSD_TRAINING_PREDICTION_HORIZON_SCHEDULE is not None else MSD_WINDOWS_PREDICTION_HORIZON),
        'stride': MSD_WINDOWS_STRIDE,
        'train_fraction': MSD_WINDOWS_TRAIN_FRACTION,
        'validation_fraction': MSD_WINDOWS_VALIDATION_FRACTION,
        'test_fraction': MSD_WINDOWS_TEST_FRACTION,
        'selection_seed': MSD_WINDOWS_SELECTION_SEED,
    },
    'training': {
        "epochs": sum(MSD_TRAINING_FIT_PHASES),
        'batch_size': MSD_TRAINING_BATCH_SIZE,
        'shuffle': MSD_TRAINING_SHUFFLE,
        'shuffle_seed': MSD_TRAINING_SHUFFLE_SEED,
        'shuffle_seed_policy': MSD_TRAINING_SHUFFLE_SEED_POLICY,
        'num_workers': MSD_TRAINING_NUM_WORKERS,
        'gradient_clip_norm': MSD_TRAINING_GRADIENT_CLIP_NORM,
        'early_stopping_patience': MSD_TRAINING_EARLY_STOPPING_PATIENCE,
        'validation_mode': MSD_TRAINING_VALIDATION_MODE,
        'validation_full_rollout_weight': MSD_TRAINING_VALIDATION_FULL_ROLLOUT_WEIGHT,
        'validation_nrmse_denominator': MSD_TRAINING_VALIDATION_NRMSE_DENOMINATOR,
        'validation_physical_metrics': MSD_TRAINING_VALIDATION_PHYSICAL_METRICS,
        'validation_skip_initial_time': MSD_TRAINING_VALIDATION_SKIP_INITIAL_TIME,
        'test_skip_initial_time': MSD_TRAINING_TEST_SKIP_INITIAL_TIME,
        'test_segment_length_time': MSD_TRAINING_TEST_SEGMENT_LENGTH_TIME,
        'drop_last': MSD_TRAINING_DROP_LAST,
        'fit_phases': MSD_TRAINING_FIT_PHASES,
        'prediction_horizon_schedule': MSD_TRAINING_PREDICTION_HORIZON_SCHEDULE,
        'state_derivative_limit_schedule': MSD_TRAINING_STATE_DERIVATIVE_LIMIT_SCHEDULE,
        'phase_shuffle_master_seed': MSD_TRAINING_PHASE_SHUFFLE_MASTER_SEED,
        'restore_history_between_phases': MSD_TRAINING_RESTORE_HISTORY_BETWEEN_PHASES,
        'reload_best_at_phase_boundary': MSD_TRAINING_RELOAD_BEST_AT_PHASE_BOUNDARY,
        'reshuffle_at_phase_boundary': MSD_TRAINING_RESHUFFLE_AT_PHASE_BOUNDARY,
        'validate_initial': MSD_TRAINING_VALIDATE_INITIAL,
        'square_outputs': MSD_TRAINING_SQUARE_OUTPUTS,
    },
    'optimizers': {
        'autograd': {
            'name': MSD_OPT_AUTOGRAD_NAME,
            'learning_rate': MSD_OPT_AUTOGRAD_LEARNING_RATE,
            'weight_decay': MSD_OPT_AUTOGRAD_WEIGHT_DECAY,
            'betas': MSD_OPT_AUTOGRAD_BETAS,
            'eps': MSD_OPT_AUTOGRAD_EPS,
            'amsgrad': MSD_OPT_AUTOGRAD_AMSGRAD,
            'learning_rate_schedule': MSD_OPT_AUTOGRAD_LEARNING_RATE_SCHEDULE,
            'scheduler_type': MSD_OPT_AUTOGRAD_SCHEDULER_TYPE,
            'scheduler_parameters': MSD_OPT_AUTOGRAD_SCHEDULER_PARAMETERS,
        },
        'mgradnet_m': {
            'name': MSD_OPT_MGRADNET_M_NAME,
            'learning_rate': MSD_OPT_MGRADNET_M_LEARNING_RATE,
            'weight_decay': MSD_OPT_MGRADNET_M_WEIGHT_DECAY,
            'betas': MSD_OPT_MGRADNET_M_BETAS,
            'eps': MSD_OPT_MGRADNET_M_EPS,
            'amsgrad': MSD_OPT_MGRADNET_M_AMSGRAD,
            'learning_rate_schedule': MSD_OPT_MGRADNET_M_LEARNING_RATE_SCHEDULE,
            'scheduler_type': MSD_OPT_MGRADNET_M_SCHEDULER_TYPE,
            'scheduler_parameters': MSD_OPT_MGRADNET_M_SCHEDULER_PARAMETERS,
        },
    },
}

# - Available options: configuration value; see description above.
CT_CONFIG = {
    'data': {
        "system": 'CT',
        'data_root': CT_DATA_DATA_ROOT,
        'dataset_filename': CT_DATA_DATASET_FILENAME,
        'generation_mode': CT_DATA_GENERATION_MODE,
        'sampling_time': CT_DATA_SAMPLING_TIME,
        'train_columns': CT_DATA_TRAIN_COLUMNS,
        'validation_columns': CT_DATA_VALIDATION_COLUMNS,
        'sampling_time_column': CT_DATA_SAMPLING_TIME_COLUMN,
        'validation_raw_samples': CT_DATA_VALIDATION_RAW_SAMPLES,
        'normalization': {
            'enabled': CT_DATA_NORMALIZATION_ENABLED,
            'normalize_input': CT_DATA_NORMALIZATION_NORMALIZE_INPUT,
            'normalize_output': CT_DATA_NORMALIZATION_NORMALIZE_OUTPUT,
            'statistics_source': CT_DATA_NORMALIZATION_STATISTICS_SOURCE,
            'epsilon': CT_DATA_NORMALIZATION_EPSILON,
            'inverse_transform_predictions': CT_DATA_NORMALIZATION_INVERSE_TRANSFORM_PREDICTIONS,
            'inverse_transform_targets': CT_DATA_NORMALIZATION_INVERSE_TRANSFORM_TARGETS,
            'physical_unit_metrics': CT_DATA_NORMALIZATION_PHYSICAL_UNIT_METRICS,
            'nrmse_denominator': CT_DATA_NORMALIZATION_NRMSE_DENOMINATOR,
        },
    },
    'model': {
        'state_dim': CT_MODEL_STATE_DIM,
        'input_dim': CT_MODEL_INPUT_DIM,
        'output_dim': CT_MODEL_OUTPUT_DIM,
        'dtype': CT_MODEL_DTYPE,
        'device': CT_MODEL_DEVICE,
        'encoder_type': CT_MODEL_ENCODER_TYPE,
        'encoder_hidden_layers': CT_MODEL_ENCODER_HIDDEN_LAYERS,
        'encoder_conv_channels': CT_MODEL_ENCODER_CONV_CHANNELS,
        'encoder_kernel_sizes': CT_MODEL_ENCODER_KERNEL_SIZES,
        'encoder_pool_sizes': CT_MODEL_ENCODER_POOL_SIZES,
        'encoder_activation': CT_MODEL_ENCODER_ACTIVATION,
        'encoder_dropout': CT_MODEL_ENCODER_DROPOUT,
        'interconnection_hidden_layers': CT_MODEL_INTERCONNECTION_HIDDEN_LAYERS,
        'dissipation_hidden_layers': CT_MODEL_DISSIPATION_HIDDEN_LAYERS,
        'port_hidden_layers': CT_MODEL_PORT_HIDDEN_LAYERS,
        'hamiltonian_hidden_layers': CT_MODEL_HAMILTONIAN_HIDDEN_LAYERS,
        'activation': CT_MODEL_ACTIVATION,
        'hamiltonian_lower_bound': CT_MODEL_HAMILTONIAN_LOWER_BOUND,
        'final_layer_initialization_scale': CT_MODEL_FINAL_LAYER_INITIALIZATION_SCALE,
        'gradnet_hidden_dim': CT_MODEL_GRADNET_HIDDEN_DIM,
        'gradnet_num_modules': CT_MODEL_GRADNET_NUM_MODULES,
        'gradnet_floor_eps': CT_MODEL_GRADNET_FLOOR_EPS,
        'gradient_scales': CT_MODEL_GRADIENT_SCALES,
        'integrator': CT_MODEL_INTEGRATOR,
        'integrator_substeps': CT_MODEL_INTEGRATOR_SUBSTEPS,
        'model_sampling_time': CT_MODEL_MODEL_SAMPLING_TIME,
        'initial_state_mode': CT_MODEL_INITIAL_STATE_MODE,
        'component_modes': CT_MODEL_COMPONENT_MODES,
        'trainable_components': CT_MODEL_TRAINABLE_COMPONENTS,
        'dissipation_factor_rank': CT_MODEL_DISSIPATION_FACTOR_RANK,
        'component_seed_offsets': CT_MODEL_COMPONENT_SEED_OFFSETS,
    },
    'windows': {
        'history_length': CT_WINDOWS_HISTORY_LENGTH,
        'prediction_horizon': (list(CT_TRAINING_PREDICTION_HORIZON_SCHEDULE.values())[-1] if CT_TRAINING_PREDICTION_HORIZON_SCHEDULE is not None else CT_WINDOWS_PREDICTION_HORIZON),
        'stride': CT_WINDOWS_STRIDE,
        'train_fraction': CT_WINDOWS_TRAIN_FRACTION,
        'validation_fraction': CT_WINDOWS_VALIDATION_FRACTION,
        'test_fraction': CT_WINDOWS_TEST_FRACTION,
        'selection_seed': CT_WINDOWS_SELECTION_SEED,
    },
    'training': {
        "epochs": sum(CT_TRAINING_FIT_PHASES),
        'batch_size': CT_TRAINING_BATCH_SIZE,
        'shuffle': CT_TRAINING_SHUFFLE,
        'shuffle_seed': CT_TRAINING_SHUFFLE_SEED,
        'shuffle_seed_policy': CT_TRAINING_SHUFFLE_SEED_POLICY,
        'num_workers': CT_TRAINING_NUM_WORKERS,
        'gradient_clip_norm': CT_TRAINING_GRADIENT_CLIP_NORM,
        'early_stopping_patience': CT_TRAINING_EARLY_STOPPING_PATIENCE,
        'validation_mode': CT_TRAINING_VALIDATION_MODE,
        'validation_full_rollout_weight': CT_TRAINING_VALIDATION_FULL_ROLLOUT_WEIGHT,
        'validation_nrmse_denominator': CT_TRAINING_VALIDATION_NRMSE_DENOMINATOR,
        'validation_physical_metrics': CT_TRAINING_VALIDATION_PHYSICAL_METRICS,
        'validation_skip_initial_time': CT_TRAINING_VALIDATION_SKIP_INITIAL_TIME,
        'test_skip_initial_time': CT_TRAINING_TEST_SKIP_INITIAL_TIME,
        'test_segment_length_time': CT_TRAINING_TEST_SEGMENT_LENGTH_TIME,
        'drop_last': CT_TRAINING_DROP_LAST,
        'fit_phases': CT_TRAINING_FIT_PHASES,
        'prediction_horizon_schedule': CT_TRAINING_PREDICTION_HORIZON_SCHEDULE,
        'state_derivative_limit_schedule': CT_TRAINING_STATE_DERIVATIVE_LIMIT_SCHEDULE,
        'phase_shuffle_master_seed': CT_TRAINING_PHASE_SHUFFLE_MASTER_SEED,
        'restore_history_between_phases': CT_TRAINING_RESTORE_HISTORY_BETWEEN_PHASES,
        'reload_best_at_phase_boundary': CT_TRAINING_RELOAD_BEST_AT_PHASE_BOUNDARY,
        'reshuffle_at_phase_boundary': CT_TRAINING_RESHUFFLE_AT_PHASE_BOUNDARY,
        'validate_initial': CT_TRAINING_VALIDATE_INITIAL,
        'square_outputs': CT_TRAINING_SQUARE_OUTPUTS,
    },
    'optimizers': {
        'autograd': {
            'name': CT_OPT_AUTOGRAD_NAME,
            'learning_rate': CT_OPT_AUTOGRAD_LEARNING_RATE,
            'weight_decay': CT_OPT_AUTOGRAD_WEIGHT_DECAY,
            'betas': CT_OPT_AUTOGRAD_BETAS,
            'eps': CT_OPT_AUTOGRAD_EPS,
            'amsgrad': CT_OPT_AUTOGRAD_AMSGRAD,
            'learning_rate_schedule': CT_OPT_AUTOGRAD_LEARNING_RATE_SCHEDULE,
            'scheduler_type': CT_OPT_AUTOGRAD_SCHEDULER_TYPE,
            'scheduler_parameters': CT_OPT_AUTOGRAD_SCHEDULER_PARAMETERS,
        },
        'mgradnet_m': {
            'name': CT_OPT_MGRADNET_M_NAME,
            'learning_rate': CT_OPT_MGRADNET_M_LEARNING_RATE,
            'weight_decay': CT_OPT_MGRADNET_M_WEIGHT_DECAY,
            'betas': CT_OPT_MGRADNET_M_BETAS,
            'eps': CT_OPT_MGRADNET_M_EPS,
            'amsgrad': CT_OPT_MGRADNET_M_AMSGRAD,
            'learning_rate_schedule': CT_OPT_MGRADNET_M_LEARNING_RATE_SCHEDULE,
            'scheduler_type': CT_OPT_MGRADNET_M_SCHEDULER_TYPE,
            'scheduler_parameters': CT_OPT_MGRADNET_M_SCHEDULER_PARAMETERS,
        },
    },
}

# ===================== 6. EXECUTION (DO NOT EDIT) =====================
def make_configuration() -> ExperimentConfig:
    """Construct the experiment directly from the selected explicit settings."""
    if SYSTEM not in ("MSD", "CT"):
        raise ValueError("SYSTEM must be MSD or CT")
    settings = MSD_CONFIG if SYSTEM == "MSD" else CT_CONFIG
    data_values = dict(settings["data"])
    data_values["normalization"] = NormalizationConfig(**data_values["normalization"])
    data = DataConfig(**data_values)
    model = ModelConfig(**settings["model"])
    windows = WindowConfig(**settings["windows"])
    training_values = dict(settings["training"])
    # Horizon ablation is fixed-horizon at each level. The study executor
    # supplies each level's horizon; a baseline curriculum must not conflict.
    if STUDY == "prediction_horizon":
        training_values["prediction_horizon_schedule"] = None
    training_values["optimizers"] = {
        method: OptimizerConfig(**settings["optimizers"][method])
        for method in METHODS
    }
    training = TrainingConfig(**training_values)
    study = StudyConfig(
        study_type=STUDY, methods=tuple(METHODS),
        method_preprocessing={m:dict(METHOD_PREPROCESSING.get(m, {})) for m in METHODS},
        seed_mode=SEED_MODE,
        manual_component_seeds=dict(MANUAL_COMPONENT_SEEDS),
        seed_master=SEED_MASTER, num_seeds=NUM_SEEDS,
        explicit_model_seeds=EXPLICIT_MODEL_SEEDS, selected_seeds=dict(SELECTED_SEEDS),
        num_shuffles=NUM_SHUFFLES, shuffle_seed_master=SHUFFLE_SEED_MASTER,
        explicit_shuffle_seeds=EXPLICIT_SHUFFLE_SEEDS,
        horizon_levels=tuple(HORIZON_LEVELS), data_size_levels=tuple(DATA_SIZE_LEVELS),
        snr_levels=tuple(SNR_LEVELS), results_root=RESULTS_ROOT,
        study_name=EXPERIMENT_TITLE, reconstruction_directory=RECONSTRUCTION_DIRECTORY,
        save_figures=SAVE_FIGURES,
        plot_options={"scales": tuple(PLOT_SCALES), "formats": tuple(PLOT_FORMATS),
                      "training_metric": PLOT_TRAINING_METRIC,
                      "test_metric": PLOT_TEST_METRIC,
                      "aggregate_learning_curves": PLOT_AGGREGATE,
                      "make_individual": PLOT_MAKE_INDIVIDUAL,
                      "make_combined": PLOT_MAKE_COMBINED,
                      "make_mean_ci": PLOT_MAKE_MEAN_CI,
                      "make_convergence": PLOT_MAKE_CONVERGENCE,
                      "show_individual_runs": PLOT_SHOW_INDIVIDUAL_RUNS,
                      "individual_run_alpha": PLOT_INDIVIDUAL_RUN_ALPHA,
                      "confidence": PLOT_CONFIDENCE,
                      "train_figure_shape": PLOT_TRAIN_FIGURE_SHAPE,
                      "rollout_figure_shape": PLOT_ROLLOUT_FIGURE_SHAPE,
                      "train_title": PLOT_TRAIN_TITLE,
                      "rollout_title": PLOT_ROLLOUT_TITLE,
                      "train_axis_labels": PLOT_TRAIN_AXIS_LABELS,
                      "rollout_axis_labels": PLOT_ROLLOUT_AXIS_LABELS,
                      "model_colors": PLOT_MODEL_COLORS,
                      "ground_truth_color": PLOT_GROUND_TRUTH_COLOR,
                      "ground_truth_linestyle": PLOT_GROUND_TRUTH_LINESTYLE,
                      "train_legend_location": PLOT_TRAIN_LEGEND_LOCATION,
                      "rollout_legend_location": PLOT_ROLLOUT_LEGEND_LOCATION,
                      "train_legend_fontsize": PLOT_TRAIN_LEGEND_FONTSIZE,
                      "rollout_legend_fontsize": PLOT_ROLLOUT_LEGEND_FONTSIZE,
                      "axis_label_fontsize": PLOT_AXIS_LABEL_FONTSIZE,
                      "tick_label_fontsize": PLOT_TICK_LABEL_FONTSIZE,
                      "title_fontsize": PLOT_TITLE_FONTSIZE,
                      "figure_dpi": PLOT_FIGURE_DPI,
                      "train_xlimits": PLOT_TRAIN_XLIMITS,
                      "train_ylimits": PLOT_TRAIN_YLIMITS,
                      "rollout_ylimits": PLOT_ROLLOUT_YLIMITS,
                      "ct_time_limits": PLOT_CT_TIME_LIMITS,
                      "msd_rollout_chunk_size": PLOT_MSD_ROLLOUT_CHUNK_SIZE,
                      "msd_show_warmup": PLOT_MSD_SHOW_WARMUP,
                      "msd_post_warmup_start_epoch": PLOT_MSD_POST_WARMUP_START_EPOCH,
                      "msd_warmup_end_epoch": PLOT_MSD_WARMUP_END_EPOCH,
                      "msd_mark_warmup": PLOT_MSD_MARK_WARMUP},
    )
    config = ExperimentConfig(data=data, model=model, windows=windows,
                              training=training, study=study)
    validate_configuration(config)
    return config


PAPER_COMPARISONS = {
    "msd_baseline": ("MSD", "baseline"),
    "ct_baseline": ("CT", "baseline"),
    "msd_snr": ("MSD", "snr"),
    "msd_training_data_size": ("MSD", "training_data_size"),
    "msd_prediction_horizon": ("MSD", "prediction_horizon"),
    "ct_prediction_horizon": ("CT", "prediction_horizon"),
}


def paper_configuration(preset: str, *, num_seeds: int = 10,
                        methods=("autograd", "mgradnet_m"),
                        levels=None, optimizers=None,
                        training_overrides=None, title=None):
    """Build a paper comparison without duplicating the training implementation.

    Overrides are applied to a freshly constructed experiment configuration.
    This function does not mutate the module's editable runner constants.
    ``optimizers`` maps method names to OptimizerConfig field overrides;
    ``training_overrides`` maps TrainingConfig field names to values.
    """
    from dataclasses import replace
    if preset not in PAPER_COMPARISONS:
        raise ValueError(f"Unknown preset {preset!r}; choose from {tuple(PAPER_COMPARISONS)}")
    system, study_type = PAPER_COMPARISONS[preset]
    if not isinstance(num_seeds, int) or not 1 <= num_seeds <= 10:
        raise ValueError("num_seeds must be between 1 and 10 for the verified paper seed list")
    methods = tuple(methods)
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("methods must be a nonempty set of distinct method names")
    allowed = ("autograd", "mgradnet_m")
    if any(m not in allowed for m in methods):
        raise ValueError("The paper presets support autograd and mgradnet_m; use the full runner for other methods")
    settings = MSD_CONFIG if system == "MSD" else CT_CONFIG
    data_values = dict(settings["data"])
    data_values["normalization"] = NormalizationConfig(**data_values["normalization"])
    training_values = dict(settings["training"])
    if study_type == "prediction_horizon":
        training_values["prediction_horizon_schedule"] = None
    optimizer_values = {m: dict(settings["optimizers"][m]) for m in methods}
    for m, changes in (optimizers or {}).items():
        if m not in optimizer_values:
            raise ValueError(f"Optimizer override for unselected method {m}")
        optimizer_values[m].update(changes)
    training_values["optimizers"] = {m: OptimizerConfig(**v) for m, v in optimizer_values.items()}
    training_values.update(training_overrides or {})
    seeds = PAPER_MSD_MODEL_SEEDS if system == "MSD" else PAPER_CT_MODEL_SEEDS
    levels = tuple(levels) if levels is not None else None
    if levels is not None and not levels:
        raise ValueError("levels must be nonempty")
    horizons = levels if study_type == "prediction_horizon" and levels is not None else ((20, 50, 100, 200) if system == "MSD" else (15, 30, 60, 120))
    sizes = levels if study_type == "training_data_size" and levels is not None else (1, 2, 5, 10, 20)
    snrs = levels if study_type == "snr" and levels is not None else (15., 35., 40., 50.)
    study = StudyConfig(
        study_type=study_type, methods=methods,
        method_preprocessing={m: {"normalize": (system == "CT" and m == "autograd"), "square_outputs": False} for m in methods},
        seed_mode="independent_component_seeds" if system == "MSD" else SEED_MODE,
        manual_component_seeds=dict(MANUAL_COMPONENT_SEEDS),
        seed_master=123 if system == "MSD" else 666,
        num_seeds=num_seeds, explicit_model_seeds=seeds[:num_seeds],
        selected_seeds={}, num_shuffles=1, shuffle_seed_master=667,
        explicit_shuffle_seeds=None, horizon_levels=tuple(horizons),
        data_size_levels=tuple(sizes), snr_levels=tuple(snrs),
        results_root=RESULTS_ROOT, study_name=title or f"paper_{preset}",
        reconstruction_directory=None, save_figures=SAVE_FIGURES,
        plot_options=make_configuration().study.plot_options,
    )
    config = ExperimentConfig(
        data=DataConfig(**data_values), model=ModelConfig(**settings["model"]),
        windows=WindowConfig(**settings["windows"]),
        training=TrainingConfig(**training_values), study=study,
    )
    validate_configuration(config)
    return config


def main() -> None:
    """Build the selected runner configuration and execute the requested study."""
    config = make_configuration()
    from oeph_training import execute_study
    return execute_study(config)


if __name__ == "__main__":
    main()

