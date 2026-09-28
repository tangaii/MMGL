# MMGL Experiment 3: True Shared–Private Disentangled Alignment

This experiment tests whether a genuinely separated shared/private representation improves ABIDE ASD diagnosis when only the shared branch receives cross-modal alignment. It follows the corrected Experiment 2 protocol and keeps the original MMGL prediction path unchanged.

The six fixed configurations are:

- `D0_baseline`: no alignment.
- `D1_global_hybrid_medium`: full token hybrid alignment, λ=0.20.
- `D2_global_hybrid_strong`: full token hybrid alignment, λ=0.50.
- `D3_shared_projector_strong`: one common 36→18 projector, λ=0.50.
- `D4_disentangled_medium`: common shared projector plus modality-specific private projectors, λ=0.20.
- `D5_disentangled_strong`: the same decomposition, λ=0.50.

D4 and D5 use shared dimension 18, private dimension 18, reconstruction weight 0.10, cross-covariance weight 0.05, private modality-classification weight 0.05, 20-epoch warmup and auxiliary weight decay 1e-4. The MMGL modules retain learning rate 0.0038 and weight decay 0.11.

The local ABIDE copy contains `processed_standard_data.csv`, `modal_feat_dict.npy` and `subject_IDs.txt`. The four input modalities are PHENO (48), ANAT (6), FUNC (10) and Correlation (256). The model receives final tokens with shape `[N,4,36]`; the original flatten → OutputLayer → GraphLearn → GCN path is used for diagnosis in every configuration.

The reported run uses `/root/venvs/111/bin/python`, 32 CPU cores, 128 GB RAM and two 96 GB GPUs. The requested `/root/venvs/mmgl` environment was absent. The complete command is:

```bash
./run_all.sh
```

The script launches at most four independent jobs and at most two jobs per GPU. Set `PYTHON_BIN`, `NEPOCH` or `EARLY` only to reproduce a different execution environment; the reported run uses the defaults `1000` and `50` for the latter two.

The final artifacts are `results/fold_results.csv`, `results/oof_predictions.csv`, `results/seed_summary.csv`, `results/summary.csv` and `results/report.md`. Eighteen config-seed logs are kept under `logs/`. Temporary per-job CSV files under `results/raw/` are removed after aggregation.

This directory contains its own source and data files and does not import `alignment_validation` or `experiment2_selective_alignment`.
