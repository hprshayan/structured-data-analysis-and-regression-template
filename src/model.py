from dataclasses import dataclass
from typing import Callable, Literal, Protocol
from enum import Enum
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

import numpy

from src.utils import CategoricalEncoder, double_dim_converter, composite_function


class DataTransformer(Protocol):
    "represents the the dataset transformer"

    def fit(self, data: numpy.ndarray) -> None:
        "fit the transformer model"
        ...

    def transform(self, data: numpy.ndarray) -> numpy.ndarray:
        "transforms the input data"
        ...

    def inv_transform(self, data: numpy.ndarray) -> numpy.ndarray:
        "inversely transforms the data"
        ...


class STDScaler:
    def __init__(self) -> None:
        self._transformer: sklearn.preprocessing._data.StandardScaler = StandardScaler()

    def fit(self, data: numpy.ndarray) -> None:
        self._transformer.fit(data)

    def fit_transform(self, data: numpy.ndarray) -> numpy.ndarray:
        return self._transformer.fit_transform(data)

    def transform(self, data: numpy.ndarray) -> numpy.ndarray:
        return self._transformer.transform(data)

    def inv_transform(self, data: numpy.ndarray) -> numpy.ndarray:
        return self._transformer.inverse_transform(data)


class ModelType(Enum):
    LINEAR = "Linear"
    RANDOM_FORREST = "Random Forrest"
    RIDGE = "Ridge"
    LASSO = "Lasso"
    MLP = "Multi Layer Perceptron"


class Model(Protocol):
    "represents a machine learning model"

    def fit(self, x: numpy.ndarray, y: numpy.ndarray) -> None:
        "represents fitting the model"
        ...

    def transform(self, x: numpy.ndarray) -> numpy.ndarray:
        "represents the transform function"
        ...


class ModelPipeline(Protocol):
    "represents an end to end pipeline of a model"

    def fit(
        self,
        feature: numpy.ndarray,
        target: numpy.ndarray,
        hparams: dict[str, int | float] = dict(),
    ) -> None:
        "fits the pipeline"
        ...

    def forward(self, feature: numpy.ndarray) -> numpy.ndarray:
        "calculates the forward path"
        ...


class Pipeline:
    feature_scaler: DataTransformer
    target_scaler: DataTransformer
    categorical_encoder: CategoricalEncoder

    def __init__(self, model: Model, type_: ModelType) -> None:
        self._type = type_
        self._model = model
        self._pipeline: Callable[[numpy.ndarray], numpy.ndarray]
        self._best_hparams: dict[str, int | float | str] | None = None

    def fit(
        self,
        features: numpy.ndarray,
        targets: numpy.ndarray,
        hparams: dict[str, int | float | str] = {},
        verbose: Literal[0, 1, 2, 3] = 3,
    ) -> None:
        grid_search = GridSearchCV(self._model(), hparams, verbose=verbose)
        transformed_features = self.feature_scaler.transform(features)
        transformed_targets = self.target_scaler.transform(targets)
        if self._type != ModelType.LINEAR:
            grid_search.fit(transformed_features, transformed_targets)
            self._best_hparams = grid_search.best_params_
            self._trained_model = self._model(**self._best_hparams)
        else:
            self._trained_model = self._model()
        self._trained_model.fit(transformed_features, transformed_targets)
        self._pipeline = composite_function(
            self.target_scaler.inv_transform,
            double_dim_converter,
            self._trained_model.predict,
            self.feature_scaler.transform,
            self.categorical_encoder.encode,
        )

    def forward(self, features: numpy.ndarray) -> numpy.ndarray:
        return self._pipeline(features)

    @property
    def type(self):
        return self._type

    @property
    def best_hparams(self):
        return self._best_hparams

    @property
    def trained_model(self):
        return self._trained_model


@dataclass
class GridSearchScenario:
    model_type: ModelType
    model: Model
    hparams: dict[str, int | float | str]
