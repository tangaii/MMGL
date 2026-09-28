# MMGL Experiment 3: True Shared–Private Disentangled Alignment

## 1. Executive Summary

This experiment tests whether a genuinely disentangled shared/private representation improves ASD diagnosis when only the shared representation is aligned. D4 and D5 use independent projection spaces, reconstruction, cross-covariance decorrelation and private modality classification.

GLOBAL_ALIGNMENT_REPLICATED: **YES**

D4_DECOMPOSITION_WORKING: **YES**

D5_DECOMPOSITION_WORKING: **YES**

DISENTANGLED_ALIGNMENT_SUPPORTED: **PARTIAL**

## 2. Previous Findings

Experiment 2 found a positive global alignment trend, but its gate decomposition was invalid: gate means were approximately 0.5, gate standard deviations were around 1e-3, orthogonality loss was approximately 1, and raw and shared alignment metrics were nearly identical.

## 3. Research Hypothesis

Shared features should become more modality-invariant and receive cross-modal alignment. Private features should retain modality identity and complementary information while reconstructing the original MMGL token.

## 4. Dataset

ABIDE has 871 subjects: ASD 403 and NC 468. The four modalities are PHENO (48), ANAT (6), FUNC (10), and Correlation (256). The final MMGL modal token shape is `[N, 4, 36]`.

## 5. Method

```text
H_m [36]
│
├── Original MMGL path → flatten → OutputLayer → GraphLearn → GCN → diagnosis
├── Shared projector → Z_shared [18] → InfoNCE + CMMD
└── Private projector_m → Z_private [18] → reconstruction + modality classification
```

The prediction path always uses the original complete modal tokens. The auxiliary modules are used only in Step A. D4 and D5 use lambda_rec=0.10, lambda_xcov=0.05, lambda_mod=0.05, 20-epoch warmup and auxiliary weight decay 1e-4.

## 6. Why Experiment 2 Gate Failed

The observed gate statistics and orthogonality values show that shared and private branches were almost scaled copies of the same token. Experiment 3 therefore uses parameter-independent projection spaces.

## 7. Experimental Protocol

Three seeds, strict stratified outer 10-fold CV, inner 10% validation, validation ACC checkpoint selection and subject-level OOF predictions were used. Test labels were excluded from training, alignment, and checkpoint selection. Transductive test features remain available to the MMGL graph construction.

## 8. Main ASD Results

Three-seed OOF mean ± standard deviation. ASD is the positive class.

| Config | ACC | BA | ASD-AUC | ASD-F1 | ASD SEN | NC SPE |
|---|---:|---:|---:|---:|---:|---:|
| D0_baseline | 84.04±0.94 | 83.79±1.01 | 87.82±1.27 | 82.33±1.26 | 80.40±2.82 | 87.18±1.86 |
| D1_global_hybrid_medium | 84.46±0.37 | 84.24±0.24 | 88.96±0.62 | 82.87±0.08 | 81.22±1.45 | 87.25±1.94 |
| D2_global_hybrid_strong | 84.88±1.09 | 84.60±1.14 | 89.38±0.37 | 83.18±1.34 | 80.81±1.99 | 88.39±0.54 |
| D3_shared_projector_strong | 84.81±0.67 | 84.58±0.76 | 88.78±1.56 | 83.23±0.97 | 81.56±2.36 | 87.61±1.30 |
| D4_disentangled_medium | 84.96±0.75 | 84.72±0.82 | 89.01±0.46 | 83.37±1.02 | 81.56±2.17 | 87.89±1.18 |
| D5_disentangled_strong | 84.23±0.33 | 84.01±0.35 | 88.27±1.74 | 82.61±0.47 | 80.98±1.65 | 87.04±1.42 |

## 9. Global Alignment Replication

D1 and D2 replicate the medium and strong global hybrid conditions from Experiment 2. The integrity reference is D0 BA 83.79% / ASD-AUC 87.82%, D1 BA 84.24% / ASD-AUC 88.96%, and D2 BA 84.60% / ASD-AUC 89.38%; these are reference values, not tuning targets.

## 10. Did Disentanglement Actually Work?

The decomposition rule requires private probe ≥0.40, private-minus-shared probe gap ≥0.10 with private winning in at least 2/3 seeds, shared and private variation both >0.10, and reconstruction NMSE <1.0.

| Config | Shared probe | Private probe | Probe gap | Shared std | Private std | Recon NMSE | Cross covariance | Working |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D4_disentangled_medium | 0.69±0.02 | 1.00±0.00 | 0.31±0.02 | 0.53±0.00 | 0.40±0.01 | 0.17±0.01 | 0.05±0.00 | YES |
| D5_disentangled_strong | 0.62±0.02 | 1.00±0.00 | 0.38±0.02 | 0.70±0.00 | 0.41±0.00 | 0.16±0.01 | 0.05±0.00 | YES |

## 11. Alignment Quality

Raw metrics use complete held-out modal tokens. Shared metrics use the projected shared tokens. Private representations are not evaluated with cross-modal similarity as a higher-is-better objective.

| Config | Raw cos-gap | Raw R@1 | Raw CMMD | Shared cos-gap | Shared R@1 | Shared CMMD |
|---|---:|---:|---:|---:|---:|---:|
| D0_baseline | 0.07±0.01 | 0.02±0.00 | 0.38±0.01 | NA | NA | NA |
| D1_global_hybrid_medium | 0.23±0.01 | 0.10±0.01 | 0.22±0.01 | NA | NA | NA |
| D2_global_hybrid_strong | 0.36±0.02 | 0.22±0.01 | 0.14±0.01 | NA | NA | NA |
| D3_shared_projector_strong | 0.13±0.01 | 0.07±0.01 | 0.29±0.00 | 0.49±0.01 | 0.28±0.01 | 0.07±0.00 |
| D4_disentangled_medium | 0.10±0.01 | 0.04±0.00 | 0.33±0.01 | 0.26±0.00 | 0.13±0.01 | 0.08±0.01 |
| D5_disentangled_strong | 0.12±0.01 | 0.07±0.00 | 0.28±0.01 | 0.48±0.01 | 0.26±0.01 | 0.07±0.00 |

## 12. Shared vs Private Modality Information

Modality probe chance is 25%. The raw probe is computed on raw modal tokens; shared and private probes use the corresponding post-hoc representations.

| Config | Raw probe | Shared probe | Private probe |
|---|---:|---:|---:|
| D0_baseline | 0.91±0.01 | NA | NA |
| D1_global_hybrid_medium | 0.92±0.00 | NA | NA |
| D2_global_hybrid_strong | 0.94±0.01 | NA | NA |
| D3_shared_projector_strong | 0.92±0.01 | 0.62±0.02 | NA |
| D4_disentangled_medium | 0.91±0.00 | 0.69±0.02 | 1.00±0.00 |
| D5_disentangled_strong | 0.92±0.00 | 0.62±0.02 | 1.00±0.00 |

## 13. Matched Global vs Disentangled

Each planned comparison uses 10,000 paired subject-level bootstrap samples per seed. The bootstrap p value is exploratory because repeated-CV training sets overlap.

| Comparison | Seed | Δ BA | BA 95% CI | BA p | Δ AUC | AUC 95% CI | AUC p |
|---|---:|---:|---:|---:|---:|---:|---:|
| D1_global_hybrid_medium vs D0_baseline | 0 | 0.0063 | [-0.0089, 0.0212] | 0.414 | 0.0178 | [0.0026, 0.0329] | 0.021 |
| D1_global_hybrid_medium vs D0_baseline | 1 | 0.0160 | [-0.0027, 0.0345] | 0.091 | 0.0157 | [0.0011, 0.0309] | 0.034 |
| D1_global_hybrid_medium vs D0_baseline | 2 | -0.0088 | [-0.0250, 0.0076] | 0.2804 | 0.0007 | [-0.0159, 0.0181] | 0.926 |
| D2_global_hybrid_strong vs D0_baseline | 0 | 0.0217 | [0.0042, 0.0394] | 0.0142 | 0.0325 | [0.0137, 0.0514] | 0.001 |
| D2_global_hybrid_strong vs D0_baseline | 1 | 0.0087 | [-0.0087, 0.0261] | 0.3314 | 0.0108 | [-0.0051, 0.0264] | 0.1722 |
| D2_global_hybrid_strong vs D0_baseline | 2 | -0.0060 | [-0.0204, 0.0082] | 0.424 | 0.0036 | [-0.0116, 0.0188] | 0.645 |
| D3_shared_projector_strong vs D2_global_hybrid_strong | 0 | -0.0083 | [-0.0209, 0.0039] | 0.1952 | -0.0264 | [-0.0405, -0.0128] | 0 |
| D3_shared_projector_strong vs D2_global_hybrid_strong | 1 | 0.0002 | [-0.0166, 0.0167] | 0.9736 | 0.0002 | [-0.0184, 0.0190] | 0.9624 |
| D3_shared_projector_strong vs D2_global_hybrid_strong | 2 | 0.0076 | [-0.0069, 0.0223] | 0.2982 | 0.0080 | [-0.0026, 0.0195] | 0.1428 |
| D4_disentangled_medium vs D1_global_hybrid_medium | 0 | 0.0106 | [-0.0022, 0.0240] | 0.1134 | 0.0022 | [-0.0095, 0.0136] | 0.7126 |
| D4_disentangled_medium vs D1_global_hybrid_medium | 1 | -0.0060 | [-0.0234, 0.0106] | 0.4888 | -0.0025 | [-0.0164, 0.0113] | 0.7238 |
| D4_disentangled_medium vs D1_global_hybrid_medium | 2 | 0.0100 | [-0.0030, 0.0231] | 0.1398 | 0.0019 | [-0.0134, 0.0172] | 0.811 |
| D5_disentangled_strong vs D2_global_hybrid_strong | 0 | -0.0218 | [-0.0374, -0.0069] | 0.0046 | -0.0348 | [-0.0488, -0.0210] | 0 |
| D5_disentangled_strong vs D2_global_hybrid_strong | 1 | 0.0071 | [-0.0124, 0.0266] | 0.4598 | 0.0017 | [-0.0164, 0.0194] | 0.8728 |
| D5_disentangled_strong vs D2_global_hybrid_strong | 2 | -0.0031 | [-0.0167, 0.0112] | 0.6814 | -0.0002 | [-0.0109, 0.0110] | 0.959 |
| D5_disentangled_strong vs D3_shared_projector_strong | 0 | -0.0135 | [-0.0264, -0.0005] | 0.0406 | -0.0084 | [-0.0206, 0.0038] | 0.176 |
| D5_disentangled_strong vs D3_shared_projector_strong | 1 | 0.0069 | [-0.0105, 0.0243] | 0.434 | 0.0015 | [-0.0159, 0.0190] | 0.8772 |
| D5_disentangled_strong vs D3_shared_projector_strong | 2 | -0.0106 | [-0.0237, 0.0023] | 0.1066 | -0.0083 | [-0.0173, 0.0009] | 0.0788 |
| D5_disentangled_strong vs D4_disentangled_medium | 0 | -0.0170 | [-0.0311, -0.0030] | 0.0158 | -0.0222 | [-0.0337, -0.0109] | 0 |
| D5_disentangled_strong vs D4_disentangled_medium | 1 | 0.0058 | [-0.0119, 0.0238] | 0.5228 | -0.0008 | [-0.0161, 0.0148] | 0.9194 |
| D5_disentangled_strong vs D4_disentangled_medium | 2 | -0.0103 | [-0.0254, 0.0046] | 0.1776 | 0.0007 | [-0.0126, 0.0143] | 0.931 |

| Comparison | Mean Δ BA | Mean Δ AUC | BA seed wins |
|---|---:|---:|---:|
| D1 vs D0 | 0.00±0.01 | 0.01±0.01 | 2/3 |
| D2 vs D0 | 0.01±0.01 | 0.02±0.02 | 2/3 |
| D3 vs D2 | -0.00±0.01 | -0.01±0.02 | 2/3 |
| D4 vs D1 | 0.00±0.01 | 0.00±0.00 | 2/3 |
| D5 vs D2 | -0.01±0.01 | -0.01±0.02 | 1/3 |
| D5 vs D3 | -0.01±0.01 | -0.01±0.01 | 1/3 |
| D5 vs D4 | -0.01±0.01 | -0.01±0.01 | 1/3 |

## 14. Projection-only Control

D3 tests a common low-dimensional projection without a private branch. D5 is compared with D3 to test whether private preservation adds value.

D3 vs D2: **NEGATIVE**. D5 vs D3 mean ΔBA = -0.0057, mean ΔAUC = -0.0051.

## 15. ASD Sensitivity Analysis

ASD sensitivity and NC specificity are reported together in the main table. Any BA change must therefore be read alongside the class-specific changes rather than interpreted from ACC alone.

## 16. Verdict

GLOBAL_ALIGNMENT_REPLICATED: **YES**

D4_DECOMPOSITION_WORKING: **YES**

D5_DECOMPOSITION_WORKING: **YES**

PROJECTION_ONLY_EFFECT: **NEGATIVE**

DISENTANGLED_ALIGNMENT_SUPPORTED: **PARTIAL**

## 17. Scientific Interpretation

Observed facts are the OOF diagnostic metrics, representation audit values, alignment metrics and paired bootstrap intervals above. Interpretation is limited to the fixed rules. A private representation with high modality probe accuracy is evidence of modality identity retention; it is not an ASD biomarker.


## 18. Next Step

Validate the fixed decomposition across another dataset or model before changing any hyperparameter.

## Reproducibility

The folder contains local ABIDE data copies, independent source files, 18 config-seed logs, and final CSV/report artifacts. Execution uses `/root/venvs/111/bin/python` on 32 CPU cores, 128 GB RAM and two 96 GB GPUs. The source environment `~/venvs/mmgl` was absent.
