# Diabetes risk estimator — frontend

A static HTML, CSS and JavaScript frontend with no framework or build step. Live predictions
are sent to the FastAPI backend, which loads `../backend/model.pkl`. The existing `model.json`
continues to power cohort-reference charts only. See the [project README](../README.md) for the
complete local setup.

## Files

| File | What it is |
| --- | --- |
| `index.html` | Page structure |
| `styles.css` | All styling, light and dark themes |
| `app.js` | Form building, backend requests, gauge, comparison table |
| `model.json` | Cohort reference forest and distribution summaries |
| `diabetes-risk-standalone.html` | Everything above inlined into one file you can double-click |
| `train_export.py` | Retrains the model and rewrites `model.json` |

## Running it

`fetch()` cannot read `model.json` from a `file://` URL. Start the backend first, then serve this
folder on port 8080:

```bash
python -m http.server 8080
# then open http://localhost:8080
```

On Windows, double-click `start-website.bat`. It starts the static server and opens the website
in your browser. Python must be installed and available on `PATH`.

## The model

The live predictor is the existing joblib-serialized scikit-learn pipeline in `../backend/model.pkl`:
a `SimpleImputer` configured with `missing_values=0`, followed by an `XGBClassifier`. The API passes
an eight-value numeric array in this order: `num_preg`, `glucose_conc`, `diastolic_bp`, `insulin`,
`bmi`, `diab_pred`, `age`, `skin`. The pickle did not preserve feature labels; this order is taken
from the existing exporter and cross-checked against the fitted imputer statistics. The backend
does not retrain or replace the model.

`model.json` contains a separate 120-tree random forest retained for the existing cohort metrics,
distributions, and reference charts. It does not generate the live probability shown in the gauge.

| Metric | Held-out (231 records) |
| --- | --- |
| Accuracy | 80.5% |
| ROC AUC | 0.852 |
| Recall | 59.3% |
| Precision | 80.0% |

The reference forest stores each tree as a flat array of nodes, one node per row:

```
[featureIndex, threshold, leftChild, rightChild, leafProbability]
```

Internal nodes carry `featureIndex >= 0`; leaves carry `-1` and the probability of class 1.
`app.js` uses this forest only for cohort-reference contrasts; the live prediction is always
requested from `POST http://localhost:8000/predict`.

### Retraining

```bash
pip install scikit-learn pandas numpy
git clone https://github.com/krishnaik06/Diabetes-Prediction.git repo
python3 train_export.py     # writes model.json
```

Changing the model shape is fine — `app.js` reads the feature order, imputation means,
metrics and distributions out of `model.json`, so only that file needs to change.

## Interface notes

- **Live scoring.** Moving any slider requests a new prediction immediately; there is no submit button.
- **Not measured.** Insulin, skinfold and blood pressure can be marked unmeasured. The frontend
  sends `null`; the backend passes the model's missing marker through its fitted imputer.
- **What is moving this estimate.** Each feature is re-scored at the cohort median while
  everything else stays put; the difference is the bar. It is an ablation, not a SHAP value,
  so the bars will not sum to the total.
- **Skinfold units.** The dataset stores `skin` in inches. The form takes millimetres and
  converts, because callipers are read in millimetres.

## Limitations worth repeating

The training data is 768 records of Pima women aged 21 and over, collected in the 1980s.
The model is wrong about one time in five and misses roughly 40% of true positives. It is a
teaching artifact, not a screening tool.
