"""Serve predictions from the exported fertilizer model."""

import hashlib
import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, FiniteFloat, StringConstraints
from typing import Annotated

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = {
    "temperature": "Temparature",
    "humidity": "Humidity",
    "moisture": "Moisture",
    "soil_type": "Soil Type",
    "crop_type": "Crop Type",
    "nitrogen": "Nitrogen",
    "potassium": "Potassium",
    "phosphorus": "Phosphorous",
}
Category = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{
        "temperature": 31, "humidity": 70, "moisture": 52, "soil_type": "Sandy",
        "crop_type": "Wheat", "nitrogen": 34, "potassium": 11, "phosphorus": 24,
    }]})

    temperature: FiniteFloat
    humidity: FiniteFloat
    moisture: FiniteFloat
    soil_type: Category
    crop_type: Category
    nitrogen: FiniteFloat
    potassium: FiniteFloat
    phosphorus: FiniteFloat


class Prediction(BaseModel):
    fertilizer: str
    probability: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    manifest = json.loads((ROOT / "logs/final_run.json").read_text())
    with (ROOT / manifest["model"]).open("rb") as model_file:
        checksum = hashlib.file_digest(model_file, "sha256").hexdigest()
        if checksum != manifest["model_sha256"]:
            raise RuntimeError("Model checksum differs from the final run manifest")
        model_file.seek(0)
        bundle = joblib.load(model_file)
    app.state.pipeline = bundle["pipeline"]
    app.state.classes = bundle["classes"]
    app.state.checksum = checksum
    yield


app = FastAPI(title="Fertilizer prediction API", lifespan=lifespan)


@app.get("/health")
def health(request: Request):
    return {"status": "ready", "model_sha256": request.app.state.checksum}


@app.post("/predict", summary="Rank the three most likely fertilizers")
def predict(observation: Observation, request: Request) -> list[Prediction]:
    features = pd.DataFrame([observation.model_dump()]).rename(columns=COLUMNS)
    state = request.app.state
    probabilities = state.pipeline.predict_proba(features)[0]
    return [
        Prediction(fertilizer=state.classes[i], probability=float(probabilities[i]))
        for i in np.argsort(-probabilities)[:3]
    ]
