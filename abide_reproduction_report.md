# ABIDE MMGL Reproduction Report

## Dataset and protocol

- Dataset: the repository-provided preprocessed ABIDE data in `data/ABIDE/`
- Subjects: 871 (`468 NC`, `403 ASD` in the paper)
- Modalities: phenotype, anatomical quality, functional quality, and 256 fMRI connectivity features
- Protocol: 10-fold stratified cross-validation
- Device: CUDA (`PPU-ZW810E`)
- Environment: `~/venvs/mmgl/bin/activate`

## Exact command

```bash
source ~/venvs/mmgl/bin/activate
cd MMGL_transductive
python main.py --datadir ../data/ --GC_mode weighted-cosine \
  --MF_mode concat --MP_mode GCN --datname ABIDE --dropout 0.35 \
  --lr 0.0038 --mode simple-2 --n_head 2 --n_hidden 18 \
  --nlayer 1 --reg 0.11
```

The only source compatibility change was making `one_hot()` create its identity matrix on the label tensor's device, required by PyTorch 2.9 CUDA indexing. Model logic and experiment parameters were unchanged.

## Fold results

| Fold | ACC (%) | AUC (%) |
|---:|---:|---:|
| 1 | 90.91 | 90.40 |
| 2 | 89.66 | 89.16 |
| 3 | 85.06 | 84.94 |
| 4 | 95.40 | 95.37 |
| 5 | 94.25 | 94.12 |
| 6 | 95.40 | 95.37 |
| 7 | 98.85 | 98.94 |
| 8 | 96.55 | 96.62 |
| 9 | 71.26 | 71.36 |
| 10 | 75.86 | 75.43 |
| **Mean** | **89.32** | **89.17** |
| Std. dev. | 8.77 | 8.82 |

## Comparison with paper Table III

| Result | ACC (%) | AUC (%) | SEN (%) | SPE (%) |
|---|---:|---:|---:|---:|
| Paper MMGL | 89.77 ± 2.72 | 89.81 ± 2.56 | 90.32 ± 4.21 | 89.30 ± 6.04 |
| This reproduction | **89.32** | **89.17** | not printed by repository script | not printed by repository script |
| Difference in mean | -0.45 pp | -0.64 pp | - | - |

The paper reports the mean and standard error, while this repository's `main.py` reports the mean and fold standard deviation for ACC/AUC. Therefore the uncertainty values are not directly comparable.

Raw output: `abide_transductive_reproduction.log`.
