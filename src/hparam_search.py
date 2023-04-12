import copy as c
from pprint import pformat

import numpy
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor

from src.constants import (
    LASSO_HPARAMS,
    MLP_HPARAMS,
    RANDOM_FORREST_HPARAMS,
    RIDGE_HPARAMS,
)
from src.metrics import Metrics, MetricTypes, calculate_err_metrics
from src.model import GridSearchScenario, ModelType, Pipeline
from src.utils import export_permutation_test_score, write_to_file


def execute_hparam_search(
    train_features: numpy.ndarray,
    test_features: numpy.ndarray,
    train_targets: numpy.ndarray,
    test_targets: numpy.ndarray,
    comparison_criteria: MetricTypes,
    path: str = "texts/model_metrics.txt",
) -> Pipeline:
    def execute(scenario: GridSearchScenario) -> tuple[Pipeline, Metrics]:
        print(
            f"performing hparam grid search for the {scenario.model_type.value} model..."
        )
        pipeline = Pipeline(scenario.model, scenario.model_type)
        pipeline.fit(
            pipeline.categorical_encoder_decoder.encode_features(train_features),
            train_targets,
            hparams=scenario.hparams,
        )
        return pipeline, calculate_err_metrics(
            test_targets, pipeline.forward(test_features)
        )

    execution_scenarios = iter(
        [
            GridSearchScenario(ModelType.LINEAR, LinearRegression, {}),
            GridSearchScenario(
                ModelType.RANDOM_FORREST, RandomForestRegressor, RANDOM_FORREST_HPARAMS
            ),
            GridSearchScenario(ModelType.RIDGE, Ridge, RIDGE_HPARAMS),
            GridSearchScenario(ModelType.LASSO, Lasso, LASSO_HPARAMS),
            GridSearchScenario(ModelType.MLP, MLPRegressor, MLP_HPARAMS),
        ]
    )

    model_metrics: dict[str, Metrics] = {}
    best_metric = 1e20

    scenario = next(execution_scenarios)
    pipeline, model_metrics[scenario.model_type.value] = execute(scenario)
    metric = getattr(
        model_metrics[scenario.model_type.value], comparison_criteria.value
    )
    best_pipeline = c.deepcopy(pipeline)
    best_metric = metric

    for scenario in execution_scenarios:
        pipeline, model_metrics[scenario.model_type.value] = execute(scenario)
        metric = getattr(
            model_metrics[scenario.model_type.value], comparison_criteria.value
        )
        if metric < best_metric:
            best_pipeline = c.deepcopy(pipeline)
            best_metric = metric

    write_to_file(pformat(model_metrics), path=path)

    print("\n************************************************************************")
    print(f"Model {best_pipeline.type.value} is chosen:")
    print(f"\thyperparameters: {best_pipeline.best_hparams}")
    print(f"\tvalidation metrics: {model_metrics[best_pipeline.type.value]}")

    features = best_pipeline.categorical_encoder_decoder.encode_features(
        numpy.concatenate((train_features, test_features), axis=0)
    )
    targets = numpy.concatenate((train_targets, test_targets), axis=0)
    export_permutation_test_score(
        best_pipeline.trained_model,
        best_pipeline.feature_scaler.transform(features),
        best_pipeline.target_scaler.transform(targets),
    )
    print("\tpermutation test score of the chosen model is exported")

    return best_pipeline
