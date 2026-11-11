# Verified paper baseline and ablation settings

The MSD and CT prediction-horizon ablation ZIP archives are the ground truth for the original experiment protocols. Shared paper baselines use **MSD PH=200** and **CT PH=60**, for both AG and MGM. The CT best-performing horizons are AG=60 and MGM=120; they are not substitutions for the common baseline.

| Parameter | MSD | CT |
| --- | --- | --- |
| Baseline horizon | 200 | 60 |
| Horizon sweep | 20, 50, 100, 200 | 15, 30, 60, 120 |
| Encoder history | 20 | 4 |
| Epochs | 70 | 700 |
| Paired model seeds | 10 | 10 |
| Validation | Full rollout | Mixed: 0.9 window + 0.1 full rollout |
| Batch size | 256 | 32 |
| Integration | Output-first RK4, one substep | Output-first RK4, one substep |
| AG Hamiltonian | MLP (16,16) | MLP (8) |
| MGM Hamiltonian gradient | 4 modules, width 4 | 2 modules, width 2 |
| Shared encoder | MLP (64,64) | MLP (64,64) |
| Shared J/R/G | MLP (8) each | MLP (8) each |
| Method normalization | AG off; MGM off | AG on; MGM off |
| AG learning rate | 0.001 | 0.01 |
| MGM learning rate | 0.01 | 0.01 |

MSD uses the recorded warm-up schedule from horizon 5 to the target horizon at epoch 5 for baseline training; the prediction-horizon sweep is fixed at each selected level in the runner. MSD PH archives record independent component seeds, a 70-epoch patience budget, and full-rollout validation. CT PH archives record mixed validation with weight 0.1, 700 epochs, a 13-phase schedule, and a learning-rate schedule of `{0: 0.01, 10: 0.005, 30: 0.001, 600: 0.0002}` for both AG and MGM. The standardized runner uses patience 700 for CT (the original CT PH archives recorded `null`, i.e. disabled early stopping). These two settings prevent early stopping within the intended 700-epoch budget, but are not identical metadata.

The MSD SNR sweep is (15, 35, 40, 50) dB; the MSD training-data-size sweep is (1, 2, 5, 10, 20) segments of 1000 samples. The six notebook presets are `msd_baseline`, `ct_baseline`, `msd_snr`, `msd_training_data_size`, `msd_prediction_horizon`, and `ct_prediction_horizon`.

The runner remains the source of truth for the executable configuration. `oeph_comparisons.ipynb` selects one preset, exposes optional seed/method/level/optimization overrides, and calls the library without reimplementing training. The full runner still permits advanced settings and reconstruction of existing results.


## Execution

Open `oeph_comparisons.ipynb`, select a preset, optionally change `NUM_SEEDS`, `METHODS`, `LEVELS`, `OPTIMIZER_OVERRIDES`, or `TRAINING_OVERRIDES`, then execute the configuration and training cells. The same presets are available programmatically through `oeph_runner.paper_configuration()`. The complete editable runner remains available for advanced changes.

**Reproduction scope:** The configuration presets have been checked against the available PH run metadata and validated as Python configuration objects; full training and numerical result equivalence have not been established. The CT PH manifests label their initialization protocol `notebook_paired`; whether the refactored component-seeding implementation is bitwise equivalent remains to be checked against actual initialized weights. The original CT PH manifests record early-stopping patience as `null`, while the standardized 700-epoch runner sets 700.
