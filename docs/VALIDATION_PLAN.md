# Research validation and operational release plan

This application is a cell-line research tool. It has not been clinically validated.
A polished interface, a high development score, or passing software tests does not establish clinical utility.

## Reproducible benchmark

Run `python scripts/validate_research.py --output research/new-unique-run`.
The runner refuses an existing directory and records the seed, processed-data fingerprint,
archive hash, candidate parameters, every row's role, prediction intervals and all test predictions.
Fit/tune/calibration/test proportions are approximately 60/10/10/20. In group splits these
are proportions of entities; pair counts vary. For the both-unseen split only pairs whose
drug and cell belong to the same role are included; cross-role pairs are explicitly excluded.
Preprocessing uses fitting rows only. Mean, Ridge, ExtraTrees, histogram boosting and a fixed
three-model ensemble are compared. The minimum tuning RMSE selects the candidate, never test scores.
The deployed FNN is not evaluated as an unseen-entity model: its training already exposed it to those entities.
Do not compare its historical pair score as though it came from these new partitions.

Candidate 90% intervals use the finite-sample split-conformal absolute-residual quantile.
Coverage is marginal under exchangeability, not individual certainty. Drug/cell grouping,
small both-unseen sets and assay shift can violate assumptions. Test coverage is reported,
never forced. These intervals are not applied to the serving checkpoint. The existing FNN's
validation and historical holdout have been used for selection, so they cannot honestly be
relabelled an untouched calibration set.

All repository data have previously been available during development. New partitions improve
methodology but do not create a genuinely independent final dataset. Do not repeatedly tune on
these reported test results. A subsequent final release needs a separately acquired, provenance-verified
study and a preregistered protocol evaluated once.

Within-cell top-five overlap and rank correlations compare only measured drugs under the same dataset.
They do not adjust for toxicity, exposure, patient covariates or assay differences. No drug is
claimed clinically superior. SHAP describes model associations, not causal drug mechanisms.

## Evidence needed from the project owner

- Original assay documentation: concentration units, target transformation, assay duration, dose ranges,
  censoring, replicate handling, study versions and licenses.
- A genuinely independent licensed drug-response study, stable drug structures/identifiers and cell identifiers,
  with compatible feature generation and no patient-identifying data.
- Curated drug-target/pathway annotations with source version, evidence level and license. API keys alone
  do not supply scientific validity and should never be committed.
- Intended research workflow, expected simultaneous users, batch size, response-time budget, hosting budget
  and retention/access policies. These determine load-test acceptance criteria and infrastructure sizing.
- For any clinical programme: clinician-led intended-use definition, authorized patient-data governance,
  ethics/regulatory assessment, clinical endpoints and prospective validation. These are not inferred from IC50.

## Operational controls and remaining gates

The optional API supports bearer authentication through `PATHDSP_API_KEY` and per-process bounded
rate limiting through `PATHDSP_REQUESTS_PER_MINUTE` (default 120 per client/minute, burst equal to limit).
Liveness is separate from model readiness. Responses carry generated request IDs, timing and no-store headers.
No raw features, predictions or credentials are logged by these controls. Health endpoints are unauthenticated
for probes. Swagger docs are protected when a key is configured. Without a key the API is local research only;
a shared bearer key is not per-user authorization. Streamlit is not protected by the API key.
Use a TLS/authenticated ingress for both app and API before exposing them to users.
Multi-worker deployments need shared ingress limits and trusted proxy configuration; in-memory limits
are not global. API controls alone are not a public deployment approval.

CI runs inference, artifact integrity, input rejection, report completeness, split leakage and control tests.
Before public operation additionally establish measured load SLOs, monitoring/alerts, dependency vulnerability
review, backup restoration rehearsal, rollback, access audit, retention policy and an incident owner.
Do not claim these gates passed until evidence exists. Research validation does not certify clinical readiness.

Fresh FNN candidates can be compared on exactly the recorded roles with
`python scripts/validate_fnn.py --directory research/new-unique-run`.
This retrains the architecture from scratch with training-only preprocessing and early stopping on tuning rows.
No candidate weights are saved into `artifacts`. Its separately recorded tuning scores can be compared with
other candidates; choosing which model to deploy still requires the independent external stage.

## Matched-cell comparison workspace

Predict provides an optional comparison of up to eight catalogue drugs for one selected cell line.
It keeps measured response and model estimate distinct, uses a logarithmic concentration display,
and calculates ratios only when the chosen reference drug is included. A ratio is a predicted
in-vitro concentration comparison, not a measure of patient progress. Comparison inputs, results
and charts are captured in the downloadable session report and evidence ZIP.
