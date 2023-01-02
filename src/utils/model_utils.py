import urllib.request

import cloudpickle
import pandas as pd
from sklearn.pipeline import Pipeline


def load_model(url: str) -> Pipeline:
    file = urllib.request.urlopen(url)
    model = cloudpickle.load(file)

    # if not model_loaded_correctly(model):
    # raise Exception("Missmatch in predictions detected!")

    return model


def model_loaded_correctly(model: Pipeline) -> bool:
    sample_input = {
        "name": "Maruti Swift Dzire VDI",
        "fuel": "Diesel",
        "seller_type": "Individual",
        "transmission": "Manual",
        "km_driven": 145500,
        "owner": "First Owner",
        "mileage": "23.4 kmpl",
        "engine": "1248 CC",
        "max_power": "74 bhp",
        "seats": 5,
        "year": 2014,
    }
    output: float = model.predict(pd.DataFrame([sample_input]))[0]
    return output == 474875.0
