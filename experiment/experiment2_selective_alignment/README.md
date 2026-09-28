# MMGL Selective Alignment Validation

This folder runs the second ABIDE experiment. It tests whether aligning a learned shared part of each MMGL modality token preserves diagnosis better than globally aligning the complete token.

The baseline is the repository MMGL transductive ABIDE model with the previous strict protocol: weighted-cosine graph, concat fusion, GCN, `simple-2`, dropout 0.35, learning rate 0.0038, weight decay 0.11, `n_head=2`, `n_hidden=18`, `nlayer=1`, validation-accuracy checkpoint selection, and a 20-epoch warmup for alignment terms.

The local `data/ABIDE/` directory contains the three required files:

- `processed_standard_data.csv`
- `modal_feat_dict.npy`
- `subject_IDs.txt`

The model receives PHENO (48), ANAT (6), FUNC (10), and Correlation (256), and exposes final tokens with shape `[N, 4, 36]`. Selective runs use a per-modality gate only for the training regularizer. The original MMGL prediction path continues to use the complete token.

The execution environment used for the reported results was `/root/venvs/111/bin/python`. The server has 32 CPU cores, 128 GB RAM, and two GPUs of approximately 96 GB each.

From this directory, run the complete experiment with:

```bash
./run_all.sh
```

`run_all.sh` launches at most four independent Python jobs, with at most two jobs per GPU. Each job writes a temporary fold CSV and OOF CSV under `results/raw/`, then `aggregate.py` creates the final files and removes those temporary inputs. To select another interpreter, use for example:

```bash
PYTHON_BIN=/path/to/python ./run_all.sh
```

The final artifacts are `results/fold_results.csv`, `results/oof_predictions.csv`, `results/seed_summary.csv`, `results/summary.csv`, and `results/report.md`. Logs are stored under `logs/`.
