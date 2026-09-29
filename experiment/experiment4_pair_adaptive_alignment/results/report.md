# MMGL Experiment 4: Diagnosis-Aware Pair-Adaptive Alignment

## 1. Executive Summary

This final MMGL alignment mechanism study tests whether six heterogeneous modality pairs should receive diagnosis-aware alignment weights instead of one equal global weight.

GLOBAL_ALIGNMENT_REPLICATED: **YES**

PAIR_WEIGHTS_NONUNIFORM: **NO**

PAIR_ASSIGNMENT_MATTERS: **NO**

PAIR_ADAPTIVE_SUPPORTED: **INCONCLUSIVE**

SPARSE_TOP3_SUPPORTED: **NO**

## 2. Previous Evidence

Global alignment was replicated in Experiments 2 and 3. Experiment 3 formed an auditable shared/private decomposition, but it did not consistently exceed global alignment. Experiment 4 therefore tests pair-specific alignment while preserving the original MMGL prediction path.

## 3. Dataset and Modalities

ABIDE contains 871 subjects, including ASD=403 and NC=468. PHENO has 48 features, ANAT contains 6 anatomical QC features, FUNC contains 10 functional QC features, and Correlation contains 256 fMRI functional-connectivity-derived features. The MMGL target token shape is `[N,4,36]`.

## 4. Six Modality Pairs

| Pair | Modalities |
|---|---|
| P0 | PHENO ↔ ANAT |
| P1 | PHENO ↔ FUNC |
| P2 | PHENO ↔ Correlation |
| P3 | ANAT ↔ FUNC |
| P4 | ANAT ↔ Correlation |
| P5 | FUNC ↔ Correlation |

## 5. Why Not Learn Free Pair Weights

A free alpha parameter optimized together with alignment loss can favor pairs that are easy to align. That identifies alignment difficulty, not diagnostic utility. Experiment 4 uses the detached cosine agreement between classification and pair alignment gradients.

## 6. Diagnosis Gradient Agreement

For pair p, `s_p = cos(∇_H L_cls, ∇_H L_align,p)`. Positive scores indicate that reducing the pair loss is locally aligned with reducing classification loss; negative scores indicate local conflict. The gradients are computed on the full modal-token tensor and detached before the actual optimization backward pass.

## 7. Constant Alignment Budget

Every applied pair weight vector is normalized to mean alpha = 1, and `L_align = 1/6 Σ alpha_p L_p`. Uniform, adaptive, shuffled, top3 and balanced sparse controls therefore share the same total alignment budget. Top3 and sparse vectors have three nonzero entries whose sum is 6.

## 8. Configurations

| Config | Alignment strategy | λ |
|---|---|---:|
| E0_baseline | none | 0.00 |
| E1_uniform_global | six equal pairs | 0.50 |
| E2_diagnosis_adaptive | gradient agreement + EMA | 0.50 |
| E3_shuffled_adaptive | adaptive weights with fixed derangement | 0.50 |
| E4_adaptive_top3 | adaptive top 3 pairs | 0.50 |
| E5_balanced_sparse_control | fixed alternating three pairs | 0.50 |

## 9. OOF Diagnostic Results

Three-seed subject-level OOF mean ± standard deviation. ASD is the positive class.

| Config | ACC | BA | ASD-AUC | ASD-F1 | ASD SEN | NC SPE |
|---|---:|---:|---:|---:|---:|---:|
| E0_baseline | 84.04±0.94 | 83.79±1.01 | 87.82±1.27 | 82.33±1.26 | 80.40±2.82 | 87.18±1.86 |
| E1_uniform_global | 84.88±1.09 | 84.60±1.14 | 89.38±0.37 | 83.18±1.34 | 80.81±1.99 | 88.39±0.54 |
| E2_diagnosis_adaptive | 85.27±0.18 | 85.05±0.21 | 88.19±2.31 | 83.76±0.30 | 82.13±1.14 | 87.96±0.86 |
| E3_shuffled_adaptive | 85.15±0.75 | 84.99±0.87 | 89.41±0.55 | 83.75±1.12 | 82.80±2.73 | 87.18±1.19 |
| E4_adaptive_top3 | 84.77±1.22 | 84.57±1.04 | 87.58±1.09 | 83.29±0.93 | 81.97±2.73 | 87.18±4.09 |
| E5_balanced_sparse_control | 84.39±1.28 | 84.16±1.24 | 88.45±1.09 | 82.79±1.30 | 81.14±1.38 | 87.18±2.06 |

## 10. Global Alignment Replication

E0 and E1 are the integrity controls for Experiment 3 D0 and D2. The reference values are E0 BA 83.79±1.01% / ASD-AUC 87.82±1.27% and E1 BA 84.60±1.14% / ASD-AUC 89.38±0.37%. Differences beyond 0.10 percentage point require protocol inspection before interpretation.

## 11. Did Pair Weights Actually Differentiate?

E2 mean pair-weight standard deviation is 0.0176 and normalized entropy is 0.9999. Nonuniformity is pre-registered at pair-weight standard deviation ≥0.08.

PAIR_WEIGHTS_NONUNIFORM: **NO**

## 12. Learned Pair Weight Structure

The frequencies describe how often a pair receives the largest or top-three effective weight under the fixed criterion. They are properties of this representation and training objective, not ASD biological biomarkers.

| Config | Pair | Weight mean±std | Score mean±std | Top1 | Top3 | Held-out cos-gap | Held-out R@1 | Held-out CMMD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| E2_diagnosis_adaptive | PHENO_ANAT | 1.02±0.01 | -0.01±0.02 | 14/30 | 30/30 | 0.4628 | 0.3244 | 0.0841 |
| E2_diagnosis_adaptive | PHENO_FUNC | 1.01±0.00 | -0.02±0.02 | 0/30 | 30/30 | 0.4813 | 0.3006 | 0.0951 |
| E2_diagnosis_adaptive | PHENO_Correlation | 0.98±0.00 | -0.08±0.02 | 0/30 | 0/30 | 0.3297 | 0.2024 | 0.2009 |
| E2_diagnosis_adaptive | ANAT_FUNC | 1.02±0.00 | 0.00±0.01 | 16/30 | 30/30 | 0.3294 | 0.2016 | 0.0898 |
| E2_diagnosis_adaptive | ANAT_Correlation | 0.98±0.00 | -0.08±0.01 | 0/30 | 0/30 | 0.2694 | 0.1408 | 0.1939 |
| E2_diagnosis_adaptive | FUNC_Correlation | 0.98±0.00 | -0.08±0.01 | 0/30 | 0/30 | 0.2663 | 0.1359 | 0.1963 |
| E3_shuffled_adaptive | PHENO_ANAT | 1.01±0.00 | -0.00±0.01 | 1/30 | 30/30 | 0.4685 | 0.3320 | 0.0810 |
| E3_shuffled_adaptive | PHENO_FUNC | 0.98±0.01 | -0.01±0.02 | 0/30 | 0/30 | 0.4992 | 0.3100 | 0.0866 |
| E3_shuffled_adaptive | PHENO_Correlation | 1.02±0.00 | -0.07±0.02 | 16/30 | 30/30 | 0.3305 | 0.2022 | 0.1922 |
| E3_shuffled_adaptive | ANAT_FUNC | 0.98±0.00 | -0.00±0.01 | 0/30 | 0/30 | 0.3197 | 0.1844 | 0.0908 |
| E3_shuffled_adaptive | ANAT_Correlation | 0.98±0.00 | -0.07±0.02 | 0/30 | 0/30 | 0.2714 | 0.1539 | 0.1905 |
| E3_shuffled_adaptive | FUNC_Correlation | 1.02±0.00 | -0.07±0.02 | 13/30 | 30/30 | 0.2733 | 0.1317 | 0.1978 |
| E4_adaptive_top3 | PHENO_ANAT | 1.94±0.37 | 0.00±0.02 | 23/30 | 29/30 | 0.5254 | 0.3747 | 0.0593 |
| E4_adaptive_top3 | PHENO_FUNC | 2.00±0.00 | -0.00±0.02 | 1/30 | 30/30 | 0.5409 | 0.3375 | 0.0703 |
| E4_adaptive_top3 | PHENO_Correlation | 0.07±0.37 | -0.07±0.03 | 1/30 | 1/30 | 0.1668 | 0.0582 | 0.3495 |
| E4_adaptive_top3 | ANAT_FUNC | 1.93±0.37 | 0.00±0.01 | 5/30 | 29/30 | 0.3736 | 0.2465 | 0.0826 |
| E4_adaptive_top3 | ANAT_Correlation | 0.07±0.36 | -0.08±0.02 | 0/30 | 1/30 | 0.1645 | 0.0507 | 0.3504 |
| E4_adaptive_top3 | FUNC_Correlation | 0.00±0.00 | -0.08±0.02 | 0/30 | 0/30 | 0.1209 | 0.0530 | 0.3479 |

## 13. Adaptive vs Uniform

E2 vs E1 is the primary pair-adaptive comparison.

## 14. Correct Assignment Control

E3 uses the same diagnosis-gradient scores and weight distribution as E2, then applies a fixed derangement. A positive E2 vs E3 result supports the value of assigning weights to the pair that produced the score.

## 15. Sparse Alignment

E4 vs E2 tests adaptive top3 against all-pair adaptive weighting. E4 vs E5 compares adaptive selection with a fixed balanced three-pair schedule. E5 vs E1 tests whether using fewer pairs alone changes performance.

## 16. Pair-specific Held-out Alignment

Held-out pair metrics are reported descriptively for the best checkpoint. E2 weight versus held-out cos-gap, R@1 and negative CMMD Spearman correlations are exploratory.

| Pair | weight vs cos-gap | weight vs R@1 | weight vs -CMMD |
|---|---:|---:|---:|
| PHENO_ANAT | 0.2169 | 0.2674 | 0.4171 |
| PHENO_FUNC | 0.1626 | -0.0530 | 0.0576 |
| PHENO_Correlation | -0.1244 | 0.0694 | -0.1070 |
| ANAT_FUNC | 0.7611 | 0.6959 | 0.6659 |
| ANAT_Correlation | -0.5306 | -0.6513 | -0.3277 |
| FUNC_Correlation | -0.4812 | -0.3559 | -0.3099 |

## 17. OOF Paired Bootstrap

Each planned comparison uses 10,000 paired subject bootstrap samples per seed. Repeated-CV training sets overlap, so bootstrap p values are exploratory evidence rather than independent clinical validation.

| Comparison | Seed | Δ BA | BA 95% CI | BA p | Δ AUC | AUC 95% CI | AUC p |
|---|---:|---:|---:|---:|---:|---:|---:|
| E1_uniform_global vs E0_baseline | 0 | 0.0217 | [0.0042, 0.0394] | 0.0142 | 0.0325 | [0.0137, 0.0514] | 0.001 |
| E1_uniform_global vs E0_baseline | 1 | 0.0087 | [-0.0087, 0.0261] | 0.3314 | 0.0108 | [-0.0051, 0.0264] | 0.1722 |
| E1_uniform_global vs E0_baseline | 2 | -0.0060 | [-0.0204, 0.0082] | 0.424 | 0.0036 | [-0.0116, 0.0188] | 0.645 |
| E2_diagnosis_adaptive vs E1_uniform_global | 0 | -0.0061 | [-0.0199, 0.0080] | 0.4082 | -0.0422 | [-0.0580, -0.0265] | 0 |
| E2_diagnosis_adaptive vs E1_uniform_global | 1 | 0.0117 | [-0.0054, 0.0292] | 0.177 | 0.0069 | [-0.0107, 0.0245] | 0.4308 |
| E2_diagnosis_adaptive vs E1_uniform_global | 2 | 0.0078 | [-0.0051, 0.0213] | 0.2542 | -0.0006 | [-0.0120, 0.0110] | 0.9154 |
| E2_diagnosis_adaptive vs E3_shuffled_adaptive | 0 | -0.0051 | [-0.0196, 0.0095] | 0.4918 | -0.0384 | [-0.0533, -0.0243] | 0 |
| E2_diagnosis_adaptive vs E3_shuffled_adaptive | 1 | 0.0080 | [-0.0066, 0.0227] | 0.29 | 0.0081 | [-0.0064, 0.0218] | 0.256 |
| E2_diagnosis_adaptive vs E3_shuffled_adaptive | 2 | -0.0010 | [-0.0157, 0.0139] | 0.8848 | -0.0064 | [-0.0182, 0.0056] | 0.2914 |
| E4_adaptive_top3 vs E2_diagnosis_adaptive | 0 | 0.0034 | [-0.0086, 0.0156] | 0.567 | 0.0096 | [-0.0034, 0.0230] | 0.1572 |
| E4_adaptive_top3 vs E2_diagnosis_adaptive | 1 | -0.0030 | [-0.0223, 0.0160] | 0.7528 | -0.0102 | [-0.0295, 0.0091] | 0.2972 |
| E4_adaptive_top3 vs E2_diagnosis_adaptive | 2 | -0.0147 | [-0.0356, 0.0059] | 0.1692 | -0.0174 | [-0.0329, -0.0021] | 0.026 |
| E4_adaptive_top3 vs E5_balanced_sparse_control | 0 | 0.0029 | [-0.0101, 0.0157] | 0.6756 | -0.0077 | [-0.0192, 0.0031] | 0.1696 |
| E4_adaptive_top3 vs E5_balanced_sparse_control | 1 | 0.0027 | [-0.0143, 0.0202] | 0.7446 | -0.0004 | [-0.0176, 0.0167] | 0.9722 |
| E4_adaptive_top3 vs E5_balanced_sparse_control | 2 | 0.0069 | [-0.0143, 0.0283] | 0.5154 | -0.0180 | [-0.0330, -0.0030] | 0.0182 |
| E5_balanced_sparse_control vs E1_uniform_global | 0 | -0.0055 | [-0.0159, 0.0048] | 0.3022 | -0.0249 | [-0.0381, -0.0121] | 0.0002 |
| E5_balanced_sparse_control vs E1_uniform_global | 1 | 0.0060 | [-0.0126, 0.0240] | 0.5132 | -0.0030 | [-0.0187, 0.0131] | 0.7394 |
| E5_balanced_sparse_control vs E1_uniform_global | 2 | -0.0137 | [-0.0291, 0.0016] | 0.0754 | -0.0001 | [-0.0126, 0.0123] | 0.971 |

| Comparison | Mean Δ BA | Mean Δ AUC | BA seed wins |
|---|---:|---:|---:|
| E1 vs E0 | 0.01±0.01 | 0.02±0.02 | 2/3 |
| E2 vs E1 | 0.00±0.01 | -0.01±0.03 | 2/3 |
| E2 vs E3 | 0.00±0.01 | -0.01±0.02 | 1/3 |
| E4 vs E2 | -0.00±0.01 | -0.01±0.01 | 1/3 |
| E4 vs E5 | 0.00±0.00 | -0.01±0.01 | 3/3 |
| E5 vs E1 | -0.00±0.01 | -0.01±0.01 | 1/3 |

## 18. ASD Sensitivity

ASD sensitivity and NC specificity are shown together with BA. This identifies whether a change reflects ASD detection, NC rejection, or both.

## 19. Verdict

GLOBAL_ALIGNMENT_REPLICATED: **YES**

PAIR_WEIGHTS_NONUNIFORM: **NO**

PAIR_ASSIGNMENT_MATTERS: **NO**

PAIR_ADAPTIVE_SUPPORTED: **INCONCLUSIVE**

SPARSE_TOP3_SUPPORTED: **NO**

## 20. Final MMGL Alignment Conclusion

在当前 ABIDE 表征和固定训练协议下，uniform global alignment 已足够；额外的 pair-adaptive 复杂度没有提供可复现的诊断收益。

FINAL_MMGL_DECISION: **TRANSFER_TO_SPROMPTGL**

Transfer strategy: **uniform global**

## 21. Next Step

将固定结论迁移到 SPromptGL，使用当前最优的 uniform global、diagnosis-adaptive 或 adaptive-top3 策略进行跨模型验证；Experiment 4 之后不再在 MMGL 上继续搜索 alignment 超参数。

## Reproducibility

The folder is self-contained and uses `/root/venvs/111/bin/python` with 32 CPU cores, 128 GB RAM and two approximately 96 GB GPUs. The original data and prior experiment folders are not modified.
