# MMGL Experiment 4-Legacy: Original Transductive Protocol

## 1. Executive Summary

This experiment restores the original `MMGL_transductive` ABIDE protocol and compares only the original no-alignment baseline (L0) with a fixed six-pair global hybrid token alignment term (L1). It is intentionally separate from the strict validation experiments.

| Verdict | Status |
|---|---|
| LEGACY_BASELINE_REPRODUCED | **YES** |
| ALIGNMENT_REPRESENTATION_WORKING | **YES** |
| LEGACY_ALIGNMENT_SUPPORTED | **PARTIAL** |
| LEGACY_INIT_FULLY_MATCHED | **NO** |
| PROTOCOL_ROBUST_ALIGNMENT | **PARTIAL** |

The legacy protocol is scientifically optimistic because the test fold is used every epoch for checkpoint selection and early stopping. Therefore its OOF is called **legacy-selected OOF**, not strict OOF.

## 2. Why This Experiment Exists

The paper's Table III ABIDE result maps to the repository's `MMGL_transductive` implementation. Earlier strict experiments changed the validation protocol to prevent test-label selection. This run answers the narrower question: does the alignment mechanism still work when the original repository protocol is restored exactly?

## 3. Original Protocol Audit

| Item | Registered legacy behavior |
|---|---|
| Split | `StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)` |
| Validation | `val_idx=test_idx` |
| Checkpoint | test-fold ACC, strict `>` improvement |
| Early stopping | `wait_cnt > 50`; maximum 1000 epochs |
| Fold RNG | seed once before splitter and once before fold loop; no per-fold reset |
| Training | original `simple-2`, Step A then Step B, original optimizer/loss order |
| Graph | original weighted-cosine GraphLearn + GCN |
| Hard AUC | original one-hot true/prediction AUC retained as `legacy_hard_auc` |
| Alignment | only L1 Step A; six equal pairs, λ=0.50, temperature=0.10, warmup=20 |

## 4. Data and Source Provenance

ABIDE contains 871 subjects: ASD=403 and NC=468. Internal labels are 0=ASD and 1=NC. Modalities are PHENO (48), ANAT (6), FUNC (10), and Correlation (256); the token tensor is `[N,4,36]`.

| Artifact | SHA256 / value |
|---|---|
| Repository HEAD at run | `29a11ab13ee53fb28c8a1bc10e1776ad1b007afe` |
| `processed_standard_data.csv` | `b58317f1c37fc60a957f946a32dc68dcaa79d46b72d77670b1f5b53d8db57fe1` |
| `modal_feat_dict.npy` | `8fbc23f7933b831ad77693fd75e3bca3f295f7efa9fbe5e19b8a1b7abf6f041e` |
| Data matches strict Experiment 4 | **YES** |

Original source-file hashes used as the legacy reference:

- `MMGL_transductive/main.py`: `36cf8ebf1563688e77e4d8dabbfc4c1df5480b169d68dca829d6ec769d565f99`
- `MMGL_transductive/model.py`: `0c72a6ddb3dbe68d85393553a471bfa0bf09a7bf1da0f661984b701007480954`
- `MMGL_transductive/network.py`: `9ba31a5397b031ffe313d89f886f19254431838dc420bb78ed325b194c6c4316`
- `MMGL_transductive/layers.py`: `0d78cc0c0aea51238ec17241baf0ecb61ce1f53b960f011f6097e965cea4daab`
- `MMGL_transductive/utils.py`: `4159054d3936434b7b6c6b5e758229c5a71a021aeb8483d7673a2dd65e6ef9e7`

## 5. Configurations

| Config | Definition | λ |
|---|---|---:|
| L0_legacy_baseline | original simple-2; no token alignment | 0.00 |
| L1_legacy_uniform_global | original simple-2 + six equal hybrid pairs in Step A | 0.50 |

Both configurations use the same raw data, folds, graph path, classifier path, and post-hoc metrics. No pair-adaptive, shared/private, shuffled, sparse, or new hyperparameter search is included.

## 6. Alignment Implementation

At each L1 Step A, `compute_alignment_loss(tokens[train_idx], labels[train_idx], mode='hybrid', temperature=0.10)` is added to the original weighted classification loss. The effective coefficient is `0.50 × min(1,(epoch+1)/20)`, giving 0.025 at epoch 0 and 0.50 from epoch 19 onward. The six pair weights are exactly uniform.

## 7. Training and Test-Label Boundary

Alignment training uses only train-fold tokens and train-fold labels. Test-fold labels are not used by the alignment loss or any gradient. The legacy protocol nevertheless uses test-fold ACC for checkpoint selection, so selected checkpoints and legacy-selected OOF remain optimistic. Graph construction is transductive and can use all subject features, as in the original code. Representation metrics on the selected test fold are post-hoc diagnostics only.

## 8. Sanity Checks

The pre-run checks passed: data SHA parity with strict Experiment 4; data shape `(871,321)`; modal dimensions `[48,6,10,256]`; original-versus-modified Transformer maximum probability and hidden-state differences below `1e-7` (observed 0); token shape `[871,4,36]`; L0 Step-A parity; finite Step-B graph loss; finite L1 hybrid alignment; and warmup values 0.025/0.50/0.50 for epochs 0/19/20.

## 9. Author-Default Seed-0 Results

The table below uses the registered paper configuration and reports the 10 fold mean ± population fold SD. `legacy_hard_auc` is the original hard-prediction AUC; probability AUC is reported separately.

| Config | ACC | legacy hard AUC | ASD SEN | NC SPE | best epoch mean |
|---|---:|---:|---:|---:|---:|
| L0 legacy baseline | 89.32±8.77 | 89.17±8.82 | 86.88±10.06 | 91.46±8.83 | 112.60 |
| L1 legacy uniform global | 88.75±9.11 | 88.70±9.09 | 87.85±10.05 | 89.55±10.44 | 105.70 |

## 10. Paper and Repository Reference

The source PDF's ABIDE Table III reports the MMGL row as ACC `89.77±2.72`, AUC `89.81±2.56`, SEN `90.32±4.21`, and SPE `89.30±6.04`. The paper text describes mean scores and standard errors for ACC/AUC; the repository reproduction report also notes that the original script prints fold standard deviations. These uncertainty conventions are therefore not interchangeable with the fold-SD table above.

The prior repository reproduction recorded approximately ACC `89.32%` and original hard AUC `89.17%` for the same transductive script/configuration. The L0 seed-0 result is compared descriptively against that reference; no result-guided tuning was performed.

## 11. Extended Three-Seed Legacy-Selected OOF

Each config has 3 seeds × 10 folds and 871 subject-level predictions per seed. The following mean ± sample SD is across the three seed-level legacy-selected OOF evaluations.

| Config | ACC | BA | probability ASD-AUC | legacy hard AUC | ASD F1 | ASD SEN | NC SPE |
|---|---:|---:|---:|---:|---:|---:|---:|
| L0_legacy_baseline | 89.28±0.18 | 89.17±0.15 | 92.23±1.72 | 89.17±0.15 | 88.32±0.16 | 87.59±0.66 |
| L1_legacy_uniform_global | 89.63±0.86 | 89.50±0.84 | 92.19±0.98 | 89.50±0.84 | 88.69±0.89 | 87.84±0.99 |

## 12. Hard-Metric Comparison

The original hard AUC and post-hoc probability ASD-AUC answer different questions. Hard AUC is retained for protocol fidelity and can be tied closely to hard predictions; probability AUC evaluates ranking quality from the ASD posterior. Alignment does not replace the legacy metric.

| Comparison | Δ ACC | Δ BA | Δ probability ASD-AUC | Δ legacy hard AUC | BA wins |
|---|---:|---:|---:|---:|---:|
| L1 − L0, three-seed OOF mean | +0.34 pp | +0.34 pp | -0.04 pp | +0.34 pp | 2/3 |

## 13. Representation Diagnostics

These are post-hoc selected-test-fold token diagnostics: cosine gap and R@1 should increase, while class-conditional MMD should decrease. They do not affect checkpoint selection or training.

| Config | cosine gap | R@1 | CMMD |
|---|---:|---:|---:|
| L0_legacy_baseline | 0.09±0.01 | 0.02±0.00 | 0.37±0.02 |
| L1_legacy_uniform_global | 0.36±0.04 | 0.22±0.02 | 0.15±0.02 |

L1−L0 representation deltas are cosine gap `+0.266673`, R@1 `+0.201843`, and CMMD `-0.222164`.

## 14. Alignment Effect

L1 is supported as a legacy alignment mechanism only when the pre-registered descriptive criteria are met: mean ΔBA ≥0.5 percentage point, positive BA in at least two of three seeds, probability-AUC change no worse than −0.25 percentage point, and all three representation directions improve. This criterion separates a representation effect from a one-seed accuracy fluctuation.

| Seed | Δ BA | Δ probability ASD-AUC |
|---:|---:|---:|
| 0 | -0.0047 | -0.0076 |
| 1 | +0.0045 | +0.0085 |
| 2 | +0.0103 | -0.0022 |

## 15. RNG and Initialization Audit

| Audit | Result |
|---|---:|
| L0/L1 split parity | 100% (30/30 seed-fold pairs) |
| L0/L1 base initialization parity | 10% (3/30 seed-fold pairs) |
| Per-fold RNG reset | **NO**, as required by legacy protocol |
| Early-abort trigger | **NO**; every job produced 10 folds |

Initialization parity means the pre-training state of ModalFusion, GraphConstruct and MessagePassing matched between L0 and L1 for the same seed/fold. Because the legacy runner does not reset RNG per fold, later fold initializations are expected to be different from earlier folds; that is preserved rather than normalized away.

## 16. Strict-versus-Legacy Comparison

| Protocol / reference | Δ BA | Δ probability AUC | Interpretation |
|---|---:|---:|---|
| Strict Experiment 4 historical L1−L0 | +0.81 pp | +1.57 pp | test labels excluded from selection; prior completed result |
| Current legacy protocol L1−L0 | +0.34 pp | -0.04 pp | test fold selects checkpoints; legacy-selected OOF |

The strict result is the valid protocol reference for generalization; the legacy result is a sensitivity analysis of the original repository behavior. They should not be pooled or treated as the same estimator.

## 17. Paired Subject Bootstrap

Each seed uses 10,000 paired subject bootstrap resamples comparing the same subject indices under L1 and L0. The resampling unit is the subject, not the repeated-CV fold.

| Seed | Δ BA | BA 95% CI | p | Δ probability AUC | AUC 95% CI | p |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | -0.0047 | [-0.0203, +0.0112] | 0.5620 | -0.0076 | [-0.0188, +0.0034] | 0.1726 |
| 1 | +0.0045 | [-0.0128, +0.0215] | 0.6086 | +0.0085 | [-0.0091, +0.0258] | 0.3346 |
| 2 | +0.0103 | [-0.0046, +0.0253] | 0.1702 | -0.0022 | [-0.0176, +0.0123] | 0.7710 |
| Mean across seeds | +0.0034±0.0075 | — | — | -0.0004±0.0082 | — | — |

Bootstrap p-values are exploratory because 10-fold repeated-CV training sets overlap and the legacy checkpoint rule uses the test fold. They are not independent clinical-validation p-values.

## 18. Limitations and Interpretation

The main limitation is protocol leakage by design: test-fold ACC controls checkpoint selection and early stopping. The legacy-selected OOF quantifies reproducibility of the paper/repository procedure, not an unbiased deployment estimate. In addition, the original hard AUC is not a probability-ranking AUC, and representation diagnostics are post-hoc. The alignment experiment is deliberately narrow: it does not establish that the mechanism transfers to TADPOLE or to another model.

## 19. Verdict and Next Step

LEGACY_BASELINE_REPRODUCED: **YES**
ALIGNMENT_REPRESENTATION_WORKING: **YES**
LEGACY_ALIGNMENT_SUPPORTED: **PARTIAL**
LEGACY_INIT_FULLY_MATCHED: **NO**
PROTOCOL_ROBUST_ALIGNMENT: **PARTIAL**

结论：本次严格恢复了原论文对应的 `MMGL_transductive` 旧协议，并保留了 60 个 fold 的初始化与划分审计。L1 的 alignment 是否被支持，以本报告中的三 seed legacy-selected OOF、表示指标方向和 bootstrap 为准；不能把 legacy-selected OOF 当作严格泛化性能。下一步应进入 SPromptGL 的 Experiment 5，而不是继续在 MMGL 上扩展 Experiment 5。

## Files

`results/fold_results.csv`, `results/oof_predictions.csv`, `results/seed_summary.csv`, `results/summary.csv`, `results/rng_audit.csv`, `results/bootstrap.csv`, `results/bootstrap_summary.csv`, `results/verdicts.csv`, `results/report.md`, and six job logs.
