import random
from dataclasses import dataclass
from enum import Enum

import numpy
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.model import ModelPipeline


class MetricTypes(Enum):
    MSE = "mse"
    MAE = "mae"
    R2_SCORE = "r2_score"


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


def depict_predicted_targets(
    pipeline: ModelPipeline,
    features: numpy.ndarray,
    targets: numpy.ndarray,
    datapoint_count: int = 100,
) -> None:
    indices = random.sample(range(features.shape[0]), datapoint_count)
    predicted_targets_sample = pipeline.forward(features[indices]).flatten()
    targets_sample = targets[indices].flatten()
    output_lines: list[str] = [
        f"{'predicted':<20}{'target':<20}{'predicted-target':<20}"
    ]
    output_lines.append(
        f"{'*************':<20}{'*************':<20}{'*************':<20}"
    )
    for predicted, target in zip(predicted_targets_sample, targets_sample):
        output_lines.append(
            f"{predicted:<20.2f}{target:<20.2f}{predicted-target:<20.2f}"
        )
    print("\n".join(output_lines))
