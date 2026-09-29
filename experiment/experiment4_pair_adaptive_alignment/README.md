# MMGL Experiment 4: Diagnosis-Aware Pair-Adaptive Alignment

Experiment 4 is the final alignment mechanism study for MMGL. It tests whether the six heterogeneous ABIDE modality pairs should receive different alignment strengths according to the agreement between their alignment gradient and the ASD classification gradient.

The prediction path remains the original MMGL path:

```text
raw features → modal tokens [N,4,36] → flatten → OutputLayer → GraphLearn → GCN → ASD prediction
```

Only the training-time alignment loss changes. There is no new projector, classifier, fusion block, graph, or GNN.

The six pairs are PHENO ↔ ANAT-QC, PHENO ↔ FUNC-QC, PHENO ↔ Correlation/FC, ANAT-QC ↔ FUNC-QC, ANAT-QC ↔ Correlation/FC, and FUNC-QC ↔ Correlation/FC. ANAT and FUNC are quality-control feature modalities; Correlation is derived from fMRI functional connectivity.

## Configurations

| Config | Method | λ | Purpose |
|---|---|---:|---|
| E0_baseline | no alignment | 0.00 | Baseline |
| E1_uniform_global | six equal pair weights | 0.50 | Uniform global control |
| E2_diagnosis_adaptive | gradient agreement + EMA | 0.50 | Core adaptive experiment |
| E3_shuffled_adaptive | adaptive weights with fixed derangement | 0.50 | Assignment control |
| E4_adaptive_top3 | adaptive weights, largest three pairs only | 0.50 | Sparse adaptive control |
| E5_balanced_sparse_control | fixed alternating three-pair schedule | 0.50 | Nonadaptive sparse control |

For E2, E3 and E4, pair scores are updated at epochs 0, 5, 10, ... using `cos(∇H L_cls, ∇H L_align,p)`. Scores are detached before the single actual training backward pass. Raw weights are `1 + 0.75 * score`, followed by normalization to mean 1. EMA beta is 0.90. E3 uses `[1,2,3,4,5,0]` as a deterministic derangement. E5 alternates `[0,2,4]` and `[1,3,5]`, assigning weight 2 to active pairs.

## Protocol

ABIDE has 871 subjects, ASD=403 and NC=468. The run uses three seeds, strict stratified outer 10-fold CV, an inner 10% validation split, validation ACC checkpoint selection, and 871 subject-level OOF predictions for every config and seed. Test labels are excluded from training, gradient agreement, pair weighting, and checkpoint selection; transductive test features can participate in graph construction.

The per-fold random seeds are `seed*100000 + fold_id*100 + 17` for model initialization and `seed*100000 + fold_id*100 + 73` for training. All six configurations initialize the same base MMGL modules in the same order. The total alignment budget is fixed because every applied pair weight vector has mean 1.

## Environment and execution

The execution interpreter is `/root/venvs/111/bin/python`. The intended server has 32 CPU cores, 128 GB RAM and two approximately 96 GB GPUs. The run uses at most four independent jobs, two per GPU, with `OMP_NUM_THREADS=4` and `MKL_NUM_THREADS=4`.

From this directory, run:

```bash
source /root/venvs/111/bin/activate
./run_all.sh
```

Optional environment variables are `PYTHON_BIN`, `NEPOCH` and `EARLY`; the registered run uses 1000 maximum epochs and 50 early stopping epochs. The folder contains all source files locally and does not import either previous experiment folder.

After successful aggregation, `run_all.sh` invokes `src/final_report.py` and prints the registered `=== MMGL EXPERIMENT 4 DONE ===` terminal summary.

Final artifacts are `results/fold_results.csv`, `results/oof_predictions.csv`, `results/seed_summary.csv`, `results/summary.csv`, `results/pair_summary.csv`, and `results/report.md`. E0/E1 integrity is checked against Experiment 3 before scientific interpretation. The paired bootstrap uses 10,000 matched subject resamples per seed for the six pre-registered comparisons; its p values are exploratory because repeated-CV training sets overlap.
