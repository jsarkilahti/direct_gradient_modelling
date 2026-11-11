# Direct Gradient Modelling for Efficient Port-Hamiltonian Neural Networks

Research code accompanying the conference paper **“Direct Gradient Modelling for Efficient Port-Hamiltonian Neural Networks”**.

**Authors:** Jussi Särkilahti, Gé Jan Ember van Otterdijk, Sadegh Ebrahimkhani, Junyi Li, Marko Hinkkanen, and Maarten Schoukens.

**Affiliations:** Aalto University, Espoo, Finland; Eindhoven University of Technology, Eindhoven, the Netherlands.

## Overview

Port-Hamiltonian neural networks (pHNNs) learn models of physical systems from input–output measurements while retaining a port-Hamiltonian structure. Their dynamics use the gradient of the Hamiltonian, which represents stored energy. Conventional models compute this gradient by automatic differentiation of a neural Hamiltonian. This work compares that approach with **direct Hamiltonian-gradient modelling** using a gradient network, avoiding differentiation of a scalar Hamiltonian during model evaluation.

The implementation supports output-error pHNN identification and comparisons on:

- **MSD:** a simulated two-mass spring–damper system, with the velocity of the outer mass as output.
- **CT:** the measured cascaded-tanks benchmark.

The primary paper comparison is between `autograd` (automatic differentiation) and `mgradnet_m` (modular gradient network). The library also exposes `simple_mgradnet_m` and `gradnet_m` variants for additional experiments. All methods use the same standard MLP-based encoder and the same MLP implementations for the shared interconnection, dissipation, and port components; the Hamiltonian-gradient representation is the central model distinction.

## Repository interface

| File | Purpose |
| --- | --- |
| `oeph_comparisons.ipynb` | Six ready-to-run paper comparisons with optional seed, method, level and optimization overrides. |
| `oeph_runner.py` | Full explicit experiment configuration and command-line entry point. |
| `EXPERIMENT_SETUP.md` | Paper settings, study configurations, and reproduction instructions. |
| `oeph_data.py`, `oeph_windowing.py` | Data preparation, model-specific preprocessing, and runtime windowing. |
| `oeph_neural_models.py`, `oeph_physics.py`, `oeph_system.py` | Neural models, dynamics, and simulation. |
| `oeph_training.py`, `oeph_evaluation.py` | Training, checkpoint selection, and evaluation. |
| `oeph_reporting.py`, `oeph_plotting.py`, `oeph_saving.py` | Reports, figures, and experiment outputs. |

The notebook is the recommended public-facing interface. **`oeph_runner.py` remains the source of the complete, explicit parametrisation.** The other `oeph_*.py` modules implement the underlying library.

## Paper comparison presets

Open `oeph_comparisons.ipynb` and select one of `msd_baseline`, `ct_baseline`, `msd_snr`, `msd_training_data_size`, `msd_prediction_horizon`, or `ct_prediction_horizon`. The notebook uses `oeph_runner.paper_configuration()` and calls the existing training implementation. The paper baselines use **MSD PH=200** and **CT PH=60**; CT best-performing horizons (**AG=60**, **MGM=120**) are distinct from the shared baseline. The verified PH archives are the reference for method-specific settings. For programmatic use, call `paper_configuration(preset, ...)` and pass its result to `oeph_training.execute_study`.

## Getting started

Use a Python environment with the dependencies imported by the source modules, including NumPy, SciPy, PyTorch, Matplotlib, and pandas. A locked environment or verified installation manifest is not supplied in this snapshot, so package versions are not asserted here.

From the repository root:

```bash
python oeph_runner.py
```

Before running, edit the configuration section at the top of `oeph_runner.py` to choose the system, study, methods, seed count, data settings, and training settings. The final `MSD_CONFIG` and `CT_CONFIG` dictionaries are assembled automatically; do not edit them directly.

The selected study can be `baseline`, `prediction_horizon`, `training_data_size`, `snr`, or `reconstruct_results`. Studies vary **one experimental factor at a time** and repeat each level over the selected seeds; they do not automatically run a full Cartesian product of factors.

### Model-specific preprocessing

`METHOD_PREPROCESSING` lets each method independently inherit or override input/output normalization and output squaring. For example:

```python
METHOD_PREPROCESSING = {
    "autograd": {"normalize": False, "square_outputs": False},
    "mgradnet_m": {"normalize": True, "square_outputs": True},
}
```

Raw time-series data are shared across methods; preprocessing and windowing are applied for each method at runtime. An empty override dictionary uses the corresponding system defaults. Such preprocessing differences should be reported explicitly when comparing models.

## Results and reproducibility

Each experiment writes to `RESULTS_ROOT / EXPERIMENT_TITLE`, with separate study-level subdirectories. A typical baseline result contains:

```text
results/
└── MSD_baseline/
    └── baseline/
        ├── report_paired.txt
        ├── report_individual_autograd.txt
        ├── report_individual_mgradnet_m.txt
        ├── runs.json
        ├── statistics.json
        ├── checkpoints/
        └── figures/
```

The reports distinguish individual-model results from paired comparisons. Saved results include checkpoint-based evaluation and experiment figures. To regenerate reports and figures without retraining, select `STUDY = "reconstruct_results"` and point `RECONSTRUCTION_DIRECTORY` to the saved study-level directory containing `runs.json`.

**Paper baseline:** The updated runner contains the baseline settings and exact model-seed lists derived from the original MSD and CT scripts. Select the system and a unique experiment title, then follow [EXPERIMENT_SETUP.md](EXPERIMENT_SETUP.md) for the one-factor comparison studies. The CT PH ablation archives determine the AG and MGM optimizer schedules used in the paper presets.

## Paper and acknowledgements

**Paper:** *Direct Gradient Modelling for Efficient Port-Hamiltonian Neural Networks*.

This work is supported in part by the **Aalto University House of AI** and in part by **ABB Oy**.

This work is funded by the **European Union (Horizon Europe, ERC, COMPLETE, 101075836)**. Views and opinions expressed are those of the authors only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.

For academic use, please cite the accompanying conference paper. Publication venue, year, DOI, and a formal BibTeX entry should be added when confirmed.

## Documentation status

`EXPERIMENT_SETUP.md` documents the paper comparison settings, source-script seed protocols, and the CT learning-rate discrepancy. `oeph_comparisons.ipynb` provides a simpler interface for reproducing those settings without exposing every internal parameter.
