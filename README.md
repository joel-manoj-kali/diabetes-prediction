# Diabetes Risk Estimator

The existing frontend is a static HTML, CSS, and JavaScript app in `frontend/`. The FastAPI
backend loads the existing `backend/model.pkl` at startup and returns predictions from that model.
The pickle is not modified or retrained.

## Model and input contract

`model.pkl` is a joblib-serialized scikit-learn `Pipeline` containing a fitted
`SimpleImputer(missing_values=0)` followed by an `XGBClassifier`. It expects eight numeric values
in this exact order:

1. `num_preg`
2. `glucose_conc`
3. `diastolic_bp`
4. `insulin`
5. `bmi`
6. `diab_pred`
7. `age`
8. `skin`

The pickle does not retain feature names. This order is defined in the existing frontend exporter
and agrees with the model's fitted imputer statistics. The frontend sends a JSON object using those
feature names; the backend assembles the ordered numeric array. Skinfold is entered in millimetres
in the UI and converted to inches before sending. The optional inputs `diastolic_bp`, `insulin`,
and `skin` may be `null`; the backend maps `null` to the model's configured missing marker, zero,
and the saved imputer supplies its fitted statistics.

## Start the backend

Open PowerShell in the project root and run:

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn main:app --reload
```

Keep this terminal open. The API is at `http://localhost:8000`. Check that the model loaded at
`http://localhost:8000/health`; interactive API documentation is at `http://localhost:8000/docs`.

## Start the frontend

Open a second PowerShell window in the project root:

```powershell
cd frontend
python -m http.server 8080
```

Alternatively, double-click `frontend/start-website.bat` from File Explorer. Open
`http://localhost:8080` in a browser. Keep this terminal open too.

## Generate a prediction

Enter measurements or select one of the example profiles. The page submits a new prediction as
the values change; there is no separate submit button. Mark insulin, diastolic blood pressure, or
skinfold as not measured when unavailable. The gauge displays the positive-class probability and
the classifier's predicted class returned by the backend.

The API accepts `POST http://localhost:8000/predict` with all eight fields, for example:

```json
{
  "num_preg": 2,
  "glucose_conc": 90,
  "diastolic_bp": 70,
  "insulin": null,
  "bmi": 27.3,
  "diab_pred": 0.085,
  "age": 22,
  "skin": 0.6692913386
}
```

The sample represents the existing low-risk profile; its 17 mm skinfold is converted to inches.
The response contains `prediction` (`0` or `1`), `label` (`negative` or `positive`), and
`positive_probability` (from 0 to 1). The frontend's cohort comparison, metrics, and driver charts
continue to use its existing `frontend/model.json`; these are reference data and do not produce the
live gauge prediction.

## Troubleshooting

- **Prediction unavailable / failed to fetch:** confirm Uvicorn is still running and
  `http://localhost:8000/health` reports `"model_loaded": true`.
- **CORS error:** serve the frontend on port 8080 as above. The backend allows `localhost:8080`
  and `127.0.0.1:8080` during development.
- **`model.pkl` not found:** keep the original model at `backend/model.pkl` and start Uvicorn with
  the working directory set to `backend`.
- **Missing Python packages or incompatible model libraries:** activate the backend environment
  and run `python -m pip install -r requirements.txt`. The tested sklearn, joblib, and XGBoost
  versions are pinned there.
- **The cohort charts do not appear:** serve the frontend over HTTP rather than opening
  `index.html` directly; `model.json` must be fetched from the same static server.

This tool is educational, not a medical test or diagnostic service.