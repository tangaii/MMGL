# MMGL Selective Alignment Validation

## 1. Executive Summary

This experiment tests whether aligning only a learned shared part of each modality token is better for ASD diagnosis than globally aligning the complete token. The primary results use 871 subject-level OOF predictions from strict transductive 10-fold CV over three independent seeds.

Global alignment status: **YES**. Over-alignment status: **NO**. Selective alignment status: **PARTIAL**.

## 2. Motivation

The first-stage experiment showed stronger alignment metrics from medium to strong hybrid alignment, while BA fell from 86.03% to 84.44%. That pattern motivates separating shared information from modality-specific information instead of forcing the complete token to agree across modalities.

## 3. Protocol Corrections

- RNG is reset independently at the start of every fold before module initialization, then reset again after all modules are initialized before the epoch loop.
- Class weights use `np.bincount(..., minlength=2)`, so weights always map to internal classes 0 and 1 in the correct order.
- DX_GROUP 1 is ASD and becomes internal class 0. All reported AUC, F1 and sensitivity metrics use ASD as the positive class; specificity is NC-specificity.

## 4. Dataset

- ABIDE: 871 subjects; ASD 403 and NC 468.
- PHENO 48, ANAT 6, FUNC 10, Correlation 256.
- Final MMGL token shape: `[N, 4, 36]`.

## 5. Method

```text
H_m
├─ original MMGL prediction path → flatten → OutputLayer → GraphLearn → GCN
└─ SelectiveSharedPrivateGate
   ├─ shared = gate × H_m → hybrid alignment
   └─ private = (1 − gate) × H_m → no cross-modal alignment
```
The original prediction feature is unchanged. The gate, orthogonality loss and balance loss are used only during ModalFusion training.

## 6. Configurations

| Config | Strategy | Alignment | λ |
|---|---|---|---:|
| C0_baseline | none | none | 0.00 |
| C1_global_pair | global | pair_nce | 0.20 |
| C2_global_hybrid_medium | global | hybrid | 0.20 |
| C3_global_hybrid_strong | global | hybrid | 0.50 |
| C4_selective_hybrid_medium | selective | hybrid | 0.20 |
| C5_selective_hybrid_strong | selective | hybrid | 0.50 |

Selective runs use λ_orth=0.02, λ_balance=0.05, target shared ratio=0.50, and 20-epoch warmup. All other MMGL hyperparameters are fixed to the previous strict experiment.

## 7. OOF Main Results

Three-seed mean ± standard deviation from subject-level OOF predictions.

| Config | ACC | BA | ASD-AUC | ASD-F1 | ASD Sensitivity | NC Specificity |
|---|---:|---:|---:|---:|---:|---:|
| C0_baseline | 84.04±0.94 | 83.79±1.01 | 87.82±1.27 | 82.33±1.26 | 80.40±2.82 | 87.18±1.86 |
| C1_global_pair | 84.77±0.37 | 84.64±0.35 | 88.89±0.69 | 83.44±0.36 | 82.96±0.14 | 86.32±0.57 |
| C2_global_hybrid_medium | 84.46±0.37 | 84.24±0.24 | 88.96±0.62 | 82.87±0.08 | 81.22±1.45 | 87.25±1.94 |
| C3_global_hybrid_strong | 84.88±1.09 | 84.60±1.14 | 89.38±0.37 | 83.18±1.34 | 80.81±1.99 | 88.39±0.54 |
| C4_selective_hybrid_medium | 84.69±0.96 | 84.57±1.08 | 88.98±0.50 | 83.36±1.32 | 82.96±2.77 | 86.18±0.62 |
| C5_selective_hybrid_strong | 83.85±0.98 | 83.61±0.94 | 88.25±0.88 | 82.17±0.97 | 80.40±0.50 | 86.82±1.42 |

## 8. Alignment Results

Raw metrics are calculated on the complete held-out modal tokens. Shared metrics are calculated on the gated shared tokens for C4 and C5.

| Config | Raw cos-gap | Raw R@1 | Raw CMMD | Shared cos-gap | Shared R@1 | Shared CMMD |
|---|---:|---:|---:|---:|---:|---:|
| C0_baseline | 0.0704±0.0061 | 0.0198±0.0013 | 0.3766±0.0101 | NA | NA | NA |
| C1_global_pair | 0.2430±0.0052 | 0.1349±0.0019 | 0.2808±0.0106 | NA | NA | NA |
| C2_global_hybrid_medium | 0.2280±0.0148 | 0.1022±0.0060 | 0.2182±0.0101 | NA | NA | NA |
| C3_global_hybrid_strong | 0.3583±0.0217 | 0.2190±0.0144 | 0.1402±0.0078 | NA | NA | NA |
| C4_selective_hybrid_medium | 0.2221±0.0098 | 0.0996±0.0038 | 0.2240±0.0046 | 0.2221±0.0098 | 0.0995±0.0039 | 0.2238±0.0046 |
| C5_selective_hybrid_strong | 0.3380±0.0283 | 0.2047±0.0185 | 0.1475±0.0104 | 0.3382±0.0283 | 0.2044±0.0188 | 0.1473±0.0104 |

## 9. Gate Analysis

A high gate means that a latent component participates more strongly in the shared alignment regularizer. It is not a biomarker attribution and is not mapped directly to a brain region.

| Config | PHENO | ANAT | FUNC | Correlation |
|---|---:|---:|---:|---:|
| C4_selective_hybrid_medium | 0.5000±0.0000 | 0.5000±0.0000 | 0.5000±0.0000 | 0.5000±0.0000 |
| C5_selective_hybrid_strong | 0.5000±0.0000 | 0.5000±0.0000 | 0.5000±0.0000 | 0.5000±0.0000 |

## 10. Planned Paired Comparisons

Each comparison uses paired OOF subjects within a seed and 10,000 paired subject-level bootstrap samples. The bootstrap p value is exploratory; repeated-CV training sets overlap.

| Comparison | Seed | Δ BA | BA 95% CI | BA p | Δ AUC | AUC 95% CI | AUC p |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1_global_pair vs C0_baseline | 0 | 0.0068 | [-0.0088, 0.0226] | 0.406 | 0.0160 | [0.0014, 0.0312] | 0.0334 |
| C1_global_pair vs C0_baseline | 1 | 0.0223 | [0.0027, 0.0424] | 0.026 | 0.0137 | [-0.0033, 0.0315] | 0.121 |
| C1_global_pair vs C0_baseline | 2 | -0.0034 | [-0.0206, 0.0137] | 0.6914 | 0.0024 | [-0.0166, 0.0212] | 0.8138 |
| C2_global_hybrid_medium vs C0_baseline | 0 | 0.0063 | [-0.0089, 0.0212] | 0.4168 | 0.0178 | [0.0026, 0.0327] | 0.0228 |
| C2_global_hybrid_medium vs C0_baseline | 1 | 0.0160 | [-0.0031, 0.0345] | 0.0976 | 0.0157 | [0.0008, 0.0308] | 0.0398 |
| C2_global_hybrid_medium vs C0_baseline | 2 | -0.0088 | [-0.0255, 0.0076] | 0.2932 | 0.0007 | [-0.0166, 0.0179] | 0.9508 |
| C3_global_hybrid_strong vs C0_baseline | 0 | 0.0217 | [0.0041, 0.0398] | 0.0158 | 0.0325 | [0.0136, 0.0519] | 0.0008 |
| C3_global_hybrid_strong vs C0_baseline | 1 | 0.0087 | [-0.0086, 0.0263] | 0.3348 | 0.0108 | [-0.0048, 0.0268] | 0.1702 |
| C3_global_hybrid_strong vs C0_baseline | 2 | -0.0060 | [-0.0204, 0.0088] | 0.4386 | 0.0036 | [-0.0115, 0.0190] | 0.6496 |
| C4_selective_hybrid_medium vs C2_global_hybrid_medium | 0 | -0.0062 | [-0.0212, 0.0089] | 0.4158 | 0.0052 | [-0.0120, 0.0229] | 0.5538 |
| C4_selective_hybrid_medium vs C2_global_hybrid_medium | 1 | -0.0020 | [-0.0203, 0.0165] | 0.8258 | -0.0088 | [-0.0228, 0.0052] | 0.223 |
| C4_selective_hybrid_medium vs C2_global_hybrid_medium | 2 | 0.0183 | [0.0025, 0.0337] | 0.0234 | 0.0044 | [-0.0081, 0.0172] | 0.503 |
| C5_selective_hybrid_strong vs C3_global_hybrid_strong | 0 | -0.0121 | [-0.0260, 0.0016] | 0.0896 | -0.0251 | [-0.0377, -0.0125] | 0.0002 |
| C5_selective_hybrid_strong vs C3_global_hybrid_strong | 1 | -0.0043 | [-0.0229, 0.0141] | 0.6452 | -0.0027 | [-0.0197, 0.0143] | 0.757 |
| C5_selective_hybrid_strong vs C3_global_hybrid_strong | 2 | -0.0133 | [-0.0309, 0.0041] | 0.127 | -0.0061 | [-0.0157, 0.0036] | 0.2078 |

| Comparison | Mean Δ BA across seeds | Mean Δ AUC across seeds | Positive BA seeds |
|---|---:|---:|---:|
| C1 vs C0 | 0.01±0.01 | 0.01±0.01 | 2/3 |
| C2 vs C0 | 0.00±0.01 | 0.01±0.01 | 2/3 |
| C3 vs C0 | 0.01±0.01 | 0.02±0.02 | 2/3 |
| C4 vs C2 | 0.00±0.01 | 0.00±0.01 | 1/3 |
| C5 vs C3 | -0.01±0.00 | -0.01±0.01 | 0/3 |

## 11. Over-Alignment Replication

C3 is compared with C2. The observed decision is **NO**. Global ΔBA (C3−C2) = 0.0036; selective ΔBA (C5−C4) = -0.0096.

## 12. Does Selective Alignment Fix It?

C4 vs C2 tests matched medium strength and C5 vs C3 tests matched strong strength. The preregistered selective rule requires mean BA gain ≥0.5 percentage points, positive BA deltas in at least 2/3 seeds, mean AUC delta ≥−0.25 percentage points, and no gate collapse.

- C4 rule: PARTIAL.
- C5 rule: FAIL.

## 13. Interpretation

Observed facts are the OOF metrics, alignment metrics, gate means and paired bootstrap intervals reported above. The statuses use the fixed rules in the experiment prompt.

Interpretation: selective alignment can support the hypothesis only when shared-token gains coincide with stable diagnosis and non-collapsed gates. Gate differences describe participation in the regularizer; they do not identify disease biology.

## 14. Verdict

GLOBAL_ALIGNMENT_REPLICATED: **YES**

OVER_ALIGNMENT_REPLICATED: **NO**

SELECTIVE_ALIGNMENT_SUPPORTED: **PARTIAL**

## 15. Next Step

Stop adding alignment terms to the four heterogeneous MMGL modalities until a selective/shared-private design is justified by a new representation analysis.

## Reproducibility

The experiment folder includes local ABIDE copies, independent source files, `run_all.sh`, and logs for all 18 config-seed jobs. The requested `~/venvs/mmgl` environment was absent; execution uses `/root/venvs/111/bin/python` on 32 CPU cores, 128 GB RAM and two GPUs.
