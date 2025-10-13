PR: Save app fixes, tuning, CI and evaluation artifacts

This PR contains the following changes:

- Robustify `app.py` so it supports CLI scoring and Streamlit interactive mode with safer feature-building.
- Added heuristic blending and tuning utilities to allow quick sensitivity experiments without retraining.
- Added tuning scripts and a precision-focused grid search (`tools/cv_and_precision_tune.py`).
- Added smoke tests (`tests/test_smoke.py`) and a GitHub Actions workflow `.github/workflows/ci.yml`.
- Added small README updates with CLI usage and a CI badge.
- Removed large runtime-generated artifacts from the repo and added `.gitignore` entries for `powerbi/out/` and `eval_out/`.

What I ran and validated locally:
- Model loads successfully and CLI scoring runs: `python app.py --csv dummy_transactions.csv`.
- Smoke test: `pytest` passed locally.
- 5-fold evaluation & precision tuning saved to `tuning_results_precision.json` and selected defaults were persisted.
- Re-scored and regenerated evaluation artifacts under `eval_out/`.

Suggested review checklist before merge:
- [ ] Confirm `tuning_results.json` defaults are acceptable for production (alpha=0.45, threshold=0.4 currently).
- [ ] Replace placeholder git user identity used for the local commit if desired.
- [ ] Optionally remove `fraud_pipeline.joblib` from the repo or move it to release artifacts (large binary in repo).
- [ ] Ensure BigQuery/Snowflake credentials and environment variables are not included in the repository.

If you want, I can:
- Open the PR for you (requires `gh` or manual browser step).
- Help retrain the model with class-weighting or sampling to improve recall.
