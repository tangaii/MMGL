# MMGL Experiment 4-Legacy: Original Transductive Protocol

This is an independent reproduction of the ABIDE experiment corresponding to the original `MMGL_transductive` implementation. It restores the original legacy behavior and compares only two configurations:

| Config | Definition | λ |
|---|---|---:|
| `L0_legacy_baseline` | Original `simple-2`, concatenation, weighted-cosine graph, GCN, no alignment | 0.00 |
| `L1_legacy_uniform_global` | Same original path plus fixed six-pair hybrid token alignment in Step A | 0.50 |

L1 uses temperature `0.10`, linear warmup `20`, and equal weight for all six modality pairs. It uses only train-fold tokens and train-fold labels for alignment. No pair-adaptive, shared/private, sparse, shuffled, or additional tuning experiment is included.

## Registered legacy protocol

- ABIDE: 871 subjects; ASD=403 and NC=468; internal label 0=ASD, 1=NC.
- Modalities: PHENO=48, ANAT=6, FUNC=10, Correlation=256; Transformer tokens `[N,4,36]`.
- `StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)`.
- `val_idx=test_idx`; test-fold ACC is used for checkpoint selection and early stopping.
- Strict `>` checkpoint improvement, `wait_cnt > 50`, maximum 1000 epochs.
- Original `simple-2` two-stage optimizer/loss order and original class-weight calculation.
- RNG is seeded once before splitter construction and once before the fold loop; it is not reset per fold.
- Original hard one-hot prediction AUC is retained as `legacy_hard_auc`; post-hoc ASD probability AUC is reported separately.

This protocol is deliberately optimistic. Its subject-level output is named **legacy-selected OOF**, not strict OOF, because the test fold selects the checkpoint.

## Provenance

The data files in this folder are byte-identical to strict Experiment 4:

```text
processed_standard_data.csv  b58317f1c37fc60a957f946a32dc68dcaa79d46b72d77670b1f5b53d8db57fe1
modal_feat_dict.npy           8fbc23f7933b831ad77693fd75e3bca3f295f7efa9fbe5e19b8a1b7abf6f041e
```

The original source reference hashes are recorded in `results/report.md`. The repository HEAD at execution time is also recorded there.

## Run

The formal run uses six fresh processes: 2 configurations × 3 seeds, at most four concurrent jobs and two jobs per GPU.

```bash
source /root/venvs/111/bin/activate
./run_all.sh
```

The default interpreter is `/root/venvs/111/bin/python`; override it with `PYTHON_BIN`. `NEPOCH`, `EARLY`, `OMP_NUM_THREADS` and `MKL_NUM_THREADS` can be overridden, but the registered run uses `NEPOCH=1000`, `EARLY=50`, and four OpenMP/MKL threads per job.

Before the formal run, execute the required checks with:

```bash
/root/venvs/111/bin/python src/sanity_check.py \
  --strict-data ../experiment4_pair_adaptive_alignment/data/ABIDE
```

## Outputs

`run_all.sh` validates six 10-fold files and six 871-row OOF files, aggregates them, removes intermediate raw per-job CSV files, and prints the final hand-off summary.

- `results/fold_results.csv`: 60 fold rows.
- `results/oof_predictions.csv`: 5,226 legacy-selected OOF rows.
- `results/seed_summary.csv`: one row per config and seed.
- `results/summary.csv`: three-seed summaries.
- `results/rng_audit.csv`: split and initial-state parity for every seed/fold.
- `results/bootstrap.csv`: 10,000 paired subject bootstrap samples summarized per seed.
- `results/bootstrap_summary.csv`: three-seed bootstrap deltas.
- `results/verdicts.csv`: registered verdicts.
- `results/report.md`: sections 1–19, paper/repository reference, strict-versus-legacy comparison, and limitations.
- `logs/*.log`: six complete job logs.

The source PDF Table III reference is MMGL ACC `89.77±2.72`, AUC `89.81±2.56`, SEN `90.32±4.21`, SPE `89.30±6.04`. The report keeps the paper's uncertainty convention separate from the fold-SD and three-seed-SD values generated here.

The next planned task after this legacy sensitivity analysis is SPromptGL Experiment 5; this folder does not start a new MMGL Experiment 5.
