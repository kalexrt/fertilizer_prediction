# Fertilizer prediction

This project ranks three of seven fertilizers for each sample in Kaggle’s **Playground Series S5E6**. It uses temperature, humidity, moisture, soil type, crop type, and nitrogen, phosphorus, and potassium levels. The final XGBoost configuration scored **0.340408 MAP@3** on a 150,000-row validation set. The trained model is included and runs on CPU through FastAPI. [Competition and data](https://www.kaggle.com/competitions/playground-series-s5e6)

## Try the model

From the project root, start the API with Docker Compose:

```bash
docker compose up --build -d
```

Open Swagger, expand **POST /predict**, select **Try it out**, and execute the example. It uses test row 750000 and returns `10-26-26`, `DAP`, and `28-28` in that order, each with its probability. **GET /health** reports readiness and the model checksum. [Swagger UI](http://localhost:8000/docs)

Use `docker compose down` to stop the service. If port 8000 is busy, set `API_PORT=8001` before the start command and open Swagger on port 8001. To run directly with Python 3.13 and uv:

```bash
uv run --group api uvicorn src.api:app
```

The container uses scikit-learn 1.6.1, matching the saved Kaggle pipeline. The local environment pins 1.5.2 and emits a version warning when loading the model.

## Test the API

From the project root, run the integration tests with Python 3.13 and uv. FastAPI’s test client loads the bundled model in-process. The sample inputs and expected rankings are included in the test file.

```bash
uv run --group api --group dev python -m pytest -q tests/test_api.py
```

- **Ranking.** For row `750000`, the test requires HTTP 200 and exactly `["10-26-26", "DAP", "28-28"]`, in submission order.
- **Validation.** The tests omit each required field in turn and exercise invalid types, null input, a blank category, an extra field, and positive and negative numeric overflow. Each case checks for HTTP 422 and the specific field and error type. Numeric fields must be finite. Physical limits such as a 100% maximum humidity are currently unenforced.
- **Concurrent requests.** Two requests send rows `750000` and `750001` in parallel. Each response must match its own saved ranking, so swapping the responses fails the test. [API integration tests](tests/test_api.py)

## Results

All comparisons use one stratified 80/20 split of the synthetic competition data. MAP@3 gives full credit for the correct fertilizer in first place, half in second, and a third in third. The same validation rows were used for model selection, so these are holdout scores.

| Model or baseline | Validation MAP@3 |
|---|---:|
| Random top three | 0.260510 |
| Three most frequent fertilizers | 0.278483 |
| Logistic regression | 0.287219 |
| LightGBM, best of six GPU configurations (depth 6, 63 leaves, seed 42) | 0.337902 |
| XGBoost, depth 6, seed 42, original round limit | 0.340220 |
| **XGBoost, depth 6, seed 42, selected 1,011 rounds** | **0.340408** |

XGBoost led tuned LightGBM at all three seeds (42, 2025, and 7), averaging **0.340507** against **0.337638**. Raising the XGBoost round limit from 1,000 to 2,000 selected 1,011 rounds by validation log loss (**1.912446**). The final refit used all **750,000 training rows** and produced three ranked labels for each of **250,000 test IDs**. [XGBoost sweep](logs/xgb_tuning_summary.json) · [LightGBM sweep](logs/lightgbm_results.csv) · [Final run and checksums](logs/final_run.json)

The original depth-6 XGBoost fit placed the correct fertilizer first on **20.34%** of validation rows and in the top three on **52.82%**. Its most common error was ranking `14-35-14` first for a true `10-26-26`. Adding a nutrient total and two ratios lowered MAP@3 by 0.006444, so the final model uses the original features. [Error analysis](logs/error_analysis.json)

## Reproduce the run

On Kaggle, attach the competition data, enable a GPU, leave `FAST_DEV` off, and run the training notebook through section 12. It selects the round count, refits the preprocessing pipeline and model, reloads the export, and writes `submission.csv` in sample-submission order. The separate LightGBM notebook runs the depth, leaf-count, and seed comparisons. [Training notebook](notebooks/fert_pred.ipynb) · [LightGBM notebook](notebooks/lightgbm.ipynb)

For a local run, put `train.csv`, `test.csv`, and `sample_submission.csv` in `data/`. Launch Jupyter from the repo root and skip the Kaggle package-install cell:

```bash
uv sync --locked
FERTILIZER_WORK_DIR="$PWD/data/local_run" uv run jupyter lab
```

The training notebook writes the model, submission, logs, and MLflow SQLite database to the run directory. Use `FAST_DEV=1` for a small smoke run. The checked-in model bundle contains the fitted pipeline and class order. [Model bundle](models/final_xgb.joblib) · [Learning curves and EDA plots](logs/figures)

Code is MIT licensed. Download the competition CSVs separately from Kaggle. [License](LICENSE)
