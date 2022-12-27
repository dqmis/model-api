import urllib.request

import cloudpickle
from sklearn.pipeline import Pipeline


def load_model(url: str) -> Pipeline:
    file = urllib.request.urlopen(url)
    model = cloudpickle.load(file)
    return model
