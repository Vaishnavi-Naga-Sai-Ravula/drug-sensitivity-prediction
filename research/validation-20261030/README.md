# Research benchmark: 30 October 2026

These experiments use only the repository data. They do not change the deployed model.

| Split | Fresh FNN RMSE | R² | Candidate 90% interval coverage | Median concentration fold error |
|---|---:|---:|---:|---:|
| pair | 0.442 | 0.864 | 90.0% | 1.80× |
| drug | 1.108 | 0.061 | 87.0% | 6.67× |
| cell | 0.595 | 0.756 | 91.2% | 2.27× |
| both | 1.121 | 0.020 | 90.1% | 6.41× |

The fresh FNN is stronger than the fixed baseline ensemble on the pair test in this run. Unseen-drug and both-unseen generalization is weak. No candidate is promoted. Scores are not directly comparable to the serving model’s historical holdout because these are different partitions and fitting fractions.

Baseline candidates include a mean predictor, Ridge, ExtraTrees, histogram gradient boosting and a fixed equal-weight ensemble. Selection uses tuning RMSE. The fresh FNN uses the same fitting/tuning/calibration/test roles, Xavier initialization, AdamW/MSE, at most 40 epochs and tuning-only early stopping. Test rows are never used for early stopping. No deployment checkpoint is replaced.

The baseline candidate selected by tuning can fail on grouped test entities. For unseen drugs, the selected histogram boosting candidate has test R² about −0.89 and interval coverage about 72%, demonstrating that nominal coverage is not guaranteed under group shift. Do not choose candidates retrospectively by the most attractive test score.

The existing repository had already been used for development. These frozen exploratory splits are not a genuinely independent final certification. A single seed does not establish stability; future grouped repeated-seed studies and a separately sourced final dataset are required.

All split membership, raw test predictions, candidate intervals, within-cell ranking metrics, file hashes, training histories and environment versions are retained here. Calibration and test groups contain few independent drug entities; row counts overstate independent biological sample size. Both-unseen excludes cross-role pairs by design.

Candidate intervals apply only to the freshly trained candidates. They are not valid serving-FNN intervals or clinical uncertainty estimates. Concentration fold errors use the supplied log transformation; μM remains conditional on the unverified source unit.

Assay provenance audit: no duplicated usable feature rows in either dataset; four CCLE feature columns are constant. CCLE has 5,703 usable responses at its minimum target value. This may reflect assay censoring, but original protocols are required before changing labels. No response rows were discarded or rescaled to improve results.

See [the validation plan](../../docs/VALIDATION_PLAN.md) for the exact scope, reproduction commands and outstanding release gates.
