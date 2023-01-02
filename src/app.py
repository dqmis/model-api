from typing import Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI
from numpy.typing import NDArray

from src.models import car, recipe
from src.utils.model_utils import load_model

# loading model
car_model = load_model("https://www.dropbox.com/s/3ov866n1p596x23/car_model.pkl?dl=1")
recipes_model = load_model("https://www.dropbox.com/s/d6apnkp38fkr2oh/recipes_model.pkl?dl=1")

app = FastAPI()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"health": "alive"}


@app.post("/predict_cars")
def predict_cars(inputs: List[car.Input]) -> Dict[str, List[Dict[str, float]]]:
    parsed_input = pd.DataFrame([i.dict() for i in inputs])
    outputs: NDArray[np.float32] = car_model.predict(parsed_input)

    return {"outputs": [{"predicted_price": i} for i in outputs.tolist()]}


@app.post("/predict_recipes")
def predict_recipes(inputs: List[recipe.Input]) -> Dict[str, List[Dict[str, float]]]:
    parsed_input = pd.DataFrame([i.dict() for i in inputs])
    outputs: NDArray[np.float32] = recipes_model.predict(parsed_input["text"].values)

    return {"outputs": [{"class": i} for i in outputs.tolist()]}
