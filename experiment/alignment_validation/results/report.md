# MMGL ABIDE alignment validation report

## 1. Executive Summary

Explicit alignment changed the held-out strict-CV results, but the predefined PROMISING_ALIGNMENT rule was not met. The best mean BA was `A4_hybrid_medium` at 86.03±10.11, compared with A0 BA 84.21±9.67; its paired BA win count was 5/10. The best mean probability AUC was `A4_hybrid_medium` at 90.14±10.87.

Original repository reproduction (transductive runner): ACC 89.32%, AUC 89.17%. The paper table reports ACC 89.77±2.72% and AUC 89.81±2.56% (paper-reported uncertainty); these are different protocols from the strict validation below.

## 2. Dataset and modalities

- ABIDE has 871 subjects. After repository label encoding (`label - 1`), class counts are 403 and 468.
- Input modalities and dimensions: `PHENO=48`, `ANAT=6`, `FUNC=10`, `Correlation=256`; total CSV input dimension is 320.
- The observed final token shape is `[N, 4, 36]` (`M=4`, `D=36`).

## 3. MMGL Integration Point

`PHENO/ANAT/FUNC/Correlation → VariLengthInputLayer → cross-modal Transformer → final per-modality modal_tokens [N,4,36] → explicit alignment regularization → original flatten/fusion (52-D hidden) → GraphLearn → GCN`.

Alignment is applied only to final per-modality tokens. The attention-specific representation is not aligned, and the original graph-learning/GCN mathematics is retained.

## 4. Alignment Objectives

- Pair InfoNCE: for every unordered modality pair, same-subject tokens are positives and other training subjects are negatives; symmetric cross-entropy is averaged and divided by `log(B)`.
- Class-conditional MMD: for each class separately, the six modality-pair distributions are compared with biased multi-RBF MMD using σ ∈ {0.25, 0.5, 1, 2}; the class losses are averaged.
- Hybrid: `L_align = 0.5 L_NCE + 0.5 L_CMMD`.
- Step A objective: `L_MF = L_cls + λ_eff L_align`, with a 20-epoch linear warmup. Alignment is train-subject-only; Step B graph/GCN training is unchanged.

## 5. Experimental Protocol

- Formal runs use stratified 10-fold outer CV. Each outer-training partition is split into 90% inner-train and 10% validation.
- Validation accuracy selects the checkpoint. Test labels are not used for checkpoint selection or alignment loss; test alignment metrics are computed only after the checkpoint is fixed.
- Fixed settings: weighted-cosine graph, concat fusion, GCN, dropout 0.35, lr 0.0038, weight decay 0.11, `simple-2`, `n_head=2`, `n_hidden=18`, `nlayer=1`.
- The requested `~/venvs/mmgl` was absent; runs used `/root/venvs/111/bin/python` with CUDA. Original source/data files were not edited.

## 6. Main Results Table

Mean ± standard deviation across available folds; summary.csv also contains standard errors.

| Config | Folds | ACC | BA | probability AUC | F1 | SEN | SPE | cos gap | R@1 | CMMD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0_baseline | 10 | 84.62±9.42 | 84.21±9.67 | 87.95±9.96 | 86.33±8.13 | 89.52±8.26 | 78.90±14.00 | 0.1070±0.0903 | 0.0234±0.0068 | 0.3412±0.0375 |
| A1_pair_nce | 10 | 85.19±8.78 | 85.07±8.84 | 90.77±9.88 | 86.23±8.23 | 86.55±9.82 | 83.59±11.27 | 0.2565±0.0628 | 0.1474±0.0307 | 0.2757±0.0404 |
| A2_cmmd | 10 | 84.61±10.11 | 84.26±10.33 | 88.81±10.30 | 86.26±8.81 | 89.11±9.13 | 79.41±14.26 | 0.1079±0.0644 | 0.0221±0.0064 | 0.1857±0.0276 |
| A3_hybrid_weak | 10 | 84.39±9.48 | 84.11±9.88 | 89.65±10.73 | 86.11±7.90 | 88.27±7.16 | 79.95±16.61 | 0.1296±0.0702 | 0.0337±0.0082 | 0.3361±0.0560 |
| A4_hybrid_medium | 10 | 86.34±9.87 | 86.03±10.11 | 90.14±10.87 | 87.74±8.55 | 89.95±8.76 | 82.11±14.28 | 0.2610±0.0901 | 0.1044±0.0207 | 0.2136±0.0514 |
| A5_hybrid_strong | 10 | 84.73±8.94 | 84.44±9.18 | 89.42±11.44 | 86.26±7.65 | 88.25±7.05 | 80.63±12.80 | 0.3361±0.0619 | 0.2009±0.0388 | 0.1452±0.0246 |

## 7. Comparison with the original paper

| Result | ACC | BA | probability AUC | SEN | SPE |
|---|---:|---:|---:|---:|---:|
| Paper MMGL ABIDE row | 89.77±2.72 | — | 89.81±2.56 | 90.32±4.21 | 89.30±6.04 |
| Original repository reproduction | 89.32±8.77 | — | 89.17±8.82 | not emitted | not emitted |
| Strict A0 baseline | 84.62±9.42 | 84.21±9.67 | 87.95±9.96 | 89.52±8.26 | 78.90±14.00 |

## 8. Paired Comparison vs Baseline

All tests use the same seed-0 outer folds. Deltas are aligned minus A0; bootstrap intervals are 95% fold-wise paired CIs. The ten-fold sample is small, so p-values are reported without suppressing non-significant results.

| Config | Metric | mean Δ | median Δ | wins | Wilcoxon p | bootstrap 95% CI |
|---|---|---:|---:|---:|---:|---:|
| A1_pair_nce | ACC | 0.0057 | 0.0000 | 5/10 | 0.6953 | [-0.0161, 0.0287] |
| A1_pair_nce | BA | 0.0086 | 0.0056 | 5/10 | 0.5566 | [-0.0152, 0.0333] |
| A1_pair_nce | prob_AUC | 0.0282 | 0.0302 | 8/10 | 0.04883 | [0.0073, 0.0484] |
| A2_cmmd | ACC | -0.0000 | -0.0115 | 3/10 | 0.6953 | [-0.0161, 0.0195] |
| A2_cmmd | BA | 0.0005 | -0.0144 | 3/10 | 0.6953 | [-0.0173, 0.0224] |
| A2_cmmd | prob_AUC | 0.0086 | 0.0011 | 5/10 | 0.5566 | [-0.0078, 0.0281] |
| A3_hybrid_weak | ACC | -0.0023 | 0.0000 | 4/10 | 0.7792 | [-0.0322, 0.0253] |
| A3_hybrid_weak | BA | -0.0010 | 0.0008 | 5/10 | 0.9056 | [-0.0325, 0.0287] |
| A3_hybrid_weak | prob_AUC | 0.0170 | 0.0154 | 7/10 | 0.1602 | [-0.0009, 0.0373] |
| A4_hybrid_medium | ACC | 0.0172 | 0.0000 | 4/10 | 0.2072 | [-0.0023, 0.0413] |
| A4_hybrid_medium | BA | 0.0182 | 0.0009 | 5/10 | 0.3743 | [-0.0025, 0.0443] |
| A4_hybrid_medium | prob_AUC | 0.0219 | 0.0192 | 9/10 | 0.04883 | [0.0054, 0.0410] |
| A5_hybrid_strong | ACC | 0.0011 | -0.0001 | 5/10 | 0.9219 | [-0.0138, 0.0172] |
| A5_hybrid_strong | BA | 0.0023 | -0.0016 | 5/10 | 1 | [-0.0134, 0.0188] |
| A5_hybrid_strong | prob_AUC | 0.0147 | 0.0066 | 7/10 | 0.1934 | [-0.0029, 0.0340] |

## 9. Alignment–Diagnosis Relationship

Config-level Spearman results use six configurations and are exploratory because n=6. For CMMD, `-CMMD` is used as the higher-is-better alignment score.

| Alignment score | Diagnosis metric | n | Spearman ρ | p |
|---|---|---:|---:|---:|
| cos_gap | BA | 6 | 0.6571 | 0.1562 |
| retrieval_r1 | BA | 6 | 0.5429 | 0.2657 |
| cmmd_score | BA | 6 | 0.4857 | 0.3287 |
| cos_gap | AUC | 6 | 0.6000 | 0.2080 |
| retrieval_r1 | AUC | 6 | 0.6000 | 0.2080 |
| cmmd_score | AUC | 6 | 0.0857 | 0.8717 |

Paired fold analysis pools aligned-config × fold deltas (5×10 = 50 observations in this run):

| Δ alignment score | Δ diagnosis | n | Spearman ρ | p |
|---|---|---:|---:|---:|
| delta_cos_gap | ΔBA | 50 | 0.1766 | 0.2198 |
| delta_cos_gap | ΔAUC | 50 | 0.1314 | 0.3631 |
| delta_retrieval_r1 | ΔBA | 50 | 0.0337 | 0.8161 |
| delta_retrieval_r1 | ΔAUC | 50 | 0.0304 | 0.8338 |
| delta_cmmd_improvement | ΔBA | 50 | -0.1730 | 0.2296 |
| delta_cmmd_improvement | ΔAUC | 50 | -0.3114 | 0.0277 |

## 10. Over-Alignment Test

| Hybrid config | BA | probability AUC | cos gap | R@1 | CMMD |
|---|---:|---:|---:|---:|---:|
| A3_hybrid_weak | 84.11±9.88 | 89.65±10.73 | 0.1296±0.0702 | 0.0337±0.0082 | 0.3361±0.0560 |
| A4_hybrid_medium | 86.03±10.11 | 90.14±10.87 | 0.2610±0.0901 | 0.1044±0.0207 | 0.2136±0.0514 |
| A5_hybrid_strong | 84.44±9.18 | 89.42±11.44 | 0.3361±0.0619 | 0.2009±0.0388 | 0.1452±0.0246 |

Observed fact: A4 medium has higher mean BA than A3 weak and A5 strong, while A5 has the strongest mean cos gap and R@1 among the three. Thus stronger alignment does not monotonically improve diagnosis in this run.

## 11. Interpretation

- Fact: A4 has the highest mean BA (86.03±10.11), and A1 has the highest mean probability AUC (90.77±9.88).
- Fact: no aligned configuration met all automatic criteria (both mean deltas ≥0.5 percentage points and at least 6/10 BA wins); A4 had 5/10 BA wins despite positive mean deltas.
- Inference: the medium hybrid may provide a useful trade-off, whereas stronger matching may suppress modality-specific complementary information. This is a hypothesis rather than a causal proof.

## 12. Verdict

`PARTIALLY_SUPPORTED` — the strict experiment shows partial mean-level evidence but does not satisfy the preregistered automatic support rule for the hypothesis: “Improving explicit multimodal alignment improves ASD diagnosis in MMGL.”

## 13. Next Step

Because the hybrid sweep shows a medium-strength optimum and stronger alignment is not monotonic, the single recommended next direction is selective/shared-private adaptive alignment that protects modality-specific information.

## Reproducibility artifacts

- `fold_results.csv`: one row per config × seed × outer fold, including explicit metric aliases and token dimensions.
- `summary.csv`: mean, standard deviation, standard error, baseline deltas, and BA/AUC win counts.
- `report.md`: this report; six config logs are under `logs/`.
- Compile check, alignment backward self-test, and one-fold paper-protocol compatibility smoke all passed before the formal runs.
