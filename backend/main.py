from contextlib import asynccontextmanager
from pathlib import Path
import logging

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


logger = logging.getLogger(__name__)
MODEL_PATH = Path(__file__).with_name("model.pkl")
FEATURE_ORDER = (
    "num_preg",
    "glucose_conc",
    "diastolic_bp",
    "insulin",
    "bmi",
    "diab_pred",
    "age",
    "skin",
)


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_preg: float = Field(allow_inf_nan=False)
    glucose_conc: float = Field(allow_inf_nan=False)
    diastolic_bp: float | None = Field(allow_inf_nan=False)
    insulin: float | None = Field(allow_inf_nan=False)
    bmi: float | None = Field(allow_inf_nan=False)
    diab_pred: float = Field(allow_inf_nan=False)
    age: float = Field(allow_inf_nan=False)
    skin: float | None = Field(allow_inf_nan=False)


def load_pipeline() -> Pipeline:
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Model file was not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    if not isinstance(model, Pipeline):
        raise RuntimeError("model.pkl must contain a scikit-learn Pipeline.")
    if int(model.n_features_in_) != len(FEATURE_ORDER):
        raise RuntimeError(
            f"Expected a model with {len(FEATURE_ORDER)} features, "
            f"but model.pkl expects {model.n_features_in_}."
        )

    imputer = model.named_steps.get("imputer")
    if not isinstance(imputer, SimpleImputer) or imputer.missing_values != 0:
        raise RuntimeError("model.pkl does not contain the expected zero-value imputer.")
    if not hasattr(model, "predict_proba") or 1 not in model.classes_:
        raise RuntimeError("model.pkl must support positive-class probability predictions.")

    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_pipeline()
    yield


app = FastAPI(title="Diabetes Risk Prediction API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "model_loaded": hasattr(request.app.state, "model")}


@app.post("/predict")
def predict(payload: PredictionInput, request: Request):
    values = [getattr(payload, feature) for feature in FEATURE_ORDER]
    row = np.asarray(
        [[0.0 if value is None else value for value in values]], dtype=np.float64
    )

    try:
        model = request.app.state.model
        prediction = int(model.predict(row)[0])
        classes = list(model.classes_)
        positive_probability = float(model.predict_proba(row)[0][classes.index(1)])
    except Exception as exc:
        logger.exception("Model prediction failed")
        raise HTTPException(
            status_code=500,
            detail="The model could not generate a prediction for these values.",
        ) from exc

    return {
        "prediction": prediction,
        "label": "positive" if prediction == 1 else "negative",
        "positive_probability": positive_probability,
    }