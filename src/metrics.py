from dataclasses import dataclass

import numpy

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class Metrics:
    "class that represents each model test error metrics"
    mse: float
    mae: float
    r2_score: float

def calculate_err_metrics(y_true: numpy.ndarray, y_pred: numpy.ndarray) -> Metrics:
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2_metric = r2_score(y_true, y_pred)
    return Metrics(mse, mae, r2_metric)
