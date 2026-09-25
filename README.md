# OpenJEV Classifier

Web app around the System-1 / System-2 hybrid classifier: single-file and folder-batch runs,
async jobs, SQLite persistence, editable taxonomy, per-file JSON/CSV + analysis.

## Run
```
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                    # + requirements-ml.txt for Qwen / Laya
python run.py                                      # http://127.0.0.1:8000
```
No models yet? `OPENJEV_CLASSIFIER_BACKEND=keyword python run.py` runs end-to-end with a dependency-free baseline.

## Layout (MVC)
- `app/config.py`       settings, dataset **profiles**, prompts - the only file you edit to adapt to a new dataset
- `app/models/`         M: SQLAlchemy entities (suppliers, jobs, file runs, row results, taxonomy versions)
- `app/controllers/`    C: one router per feature (single, batch, jobs, files, taxonomy, suppliers, system)
- `app/templates/`, `app/static/`   V: Jinja pages, CSS, small vanilla JS per page
- `app/services/`       business logic (pipeline, job manager, taxonomy, analysis, files, suppliers, diagnostics)
- `app/engine/`         classification engine (backends, hierarchical algorithm, System-2 client)
- `data/`               created at runtime: `inbox/` (batch folder), `taxonomies/<profile>.json`, `outputs/`, `logs/`, `openjev.db`

## Add a dataset
Add a `ProfileConfig` to `DEFAULT_PROFILES` in `app/config.py` (description, item name, column hints, optional seed taxonomy).
Select it in the UI; the taxonomy file `data/taxonomies/<profile>.json` is created on first use.

Smoke test (no ML needed): `python -m tests.smoke_test`
