from typing import Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI

from src.models import car
from src.utils.model_utils import load_model

# loading model
model = load_model("https://www.dropbox.com/s/3ov866n1p596x23/car_model.pkl?dl=1")

app = FastAPI()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"health": "alive"}


@app.post("/predict")
def predict(inputs: List[car.Input]) -> Dict[str, List[Dict[str, float]]]:
    parsed_input = pd.DataFrame([i.dict() for i in inputs])
    outputs: np.ndarray = model.predict(parsed_input)  # type: ignore

    return {"outputs": [{"predicted_price": i} for i in outputs.tolist()]}
