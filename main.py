"Analyzing a dataset and fitting regression models on it"

import warnings
import numpy
import copy as c
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from pprint import pformat
from sklearn.neural_network import MLPRegressor

from src.data_analysis import DatasetAnalysis
from src.metrics import MetricTypes, Metrics, calculate_err_metrics
from src.utils import (
    CategoricalEncoder,
    DatasetMetadata,
    DatasetType,
    create_directories,
    data_train_test_split,
    export_permutation_test_score,
    load_and_init_process_data,
    write_to_file,
)
from src.model import GridSearchScenario, ModelType, Pipeline, STDScaler


warnings.filterwarnings("ignore", category=DataConversionWarning)

"""
Medical Cost Personal Datasets
https://www.kaggle.com/datasets/mirichoi0218/insurance
"""
DATASET_METADATA = DatasetMetadata(
    path="insurance.csv",
    data_format=DatasetType.CSV,
    target=["charges"],
    header=0,
    skip_columns=0,
    delimiter=",",
    na_values=[],
    categorical_columns=["children", "smoker", "region"],
)

COMPARISON_CRITERIA = MetricTypes.MSE
TEST_SIZE = 0.2
SEED = 0
RANDOM_FORREST_HPARAMS: dict[str, list[any]] = {
    "max_depth": list(range(1, 20, 2)),
    "n_estimators": list(range(20, 200, 20)),
}
RIDGE_HPARAMS: dict[str, list[any]] = {"alpha": list(numpy.arange(0.1, 2.0, 0.1))}
LASSO_HPARAMS: dict[str, list[any]] = {"alpha": list(numpy.arange(0.1, 2.0, 0.1))}
MLP_HPARAMS: dict[str, list[any]] = {
    "hidden_layer_sizes": [(10,), (20,), (100,)],
    "activation": ["identity", "relu"],
    "batch_size": [1, 20, 50],
    "learning_rate": ["constant", "adaptive"],
    "learning_rate_init": [0.001, 0.01],
}


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
        pipeline.fit(train_features, train_targets, hparams=scenario.hparams)
        return pipeline, calculate_err_metrics(
            test_targets, pipeline.forward(test_features)
        )

    execution_scenarios = [
        GridSearchScenario(ModelType.LINEAR, LinearRegression, {}),
        GridSearchScenario(
            ModelType.RANDOM_FORREST, RandomForestRegressor, RANDOM_FORREST_HPARAMS
        ),
        GridSearchScenario(ModelType.RIDGE, Ridge, RIDGE_HPARAMS),
        GridSearchScenario(ModelType.LASSO, Lasso, LASSO_HPARAMS),
        GridSearchScenario(ModelType.MLP, MLPRegressor, MLP_HPARAMS),
    ]

    model_metrics: dict[str, Metrics] = {}
    best_metric = 1e5
    best_pipeline: Pipeline
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

    features = numpy.concatenate((train_features, test_features), axis=0)
    targets = numpy.concatenate((train_targets, test_targets), axis=0)
    export_permutation_test_score(
        pipeline.trained_model,
        pipeline.feature_scaler.transform(features),
        pipeline.target_scaler.transform(targets),
    )
    print("\tpermutation test score of the chosen model is exported")

    return best_pipeline


def main():
    # prepare the directories
    create_directories()

    # load the dataset, remove and save rows that contain null values and one-hot encode categorial features
    categorical_encoder = CategoricalEncoder(DATASET_METADATA)
    dataset = load_and_init_process_data(DATASET_METADATA, categorical_encoder)

    # Analyze the dataset
    print("analyzing the dataset...")
    dataset_analyzer = DatasetAnalysis(dataset, DATASET_METADATA.target)
    dataset_analyzer.analyze_dataset()

    train_features, test_features, train_targets, test_targets = data_train_test_split(
        dataset, TEST_SIZE, SEED, DATASET_METADATA.target
    )

    # fitting the dataset transformers
    feature_scaler = STDScaler()
    target_scaler = STDScaler()
    feature_scaler.fit(train_features)
    target_scaler.fit(train_targets)
    Pipeline.feature_scaler = feature_scaler
    Pipeline.target_scaler = target_scaler
    Pipeline.categorical_encoder = categorical_encoder

    # run the hyperparameter search
    print("\nexecuting hyperparameter tunning for regression models...")
    best_pipeline = execute_hparam_search(
        train_features, test_features, train_targets, test_targets, COMPARISON_CRITERIA
    )

    print("\nAll done!")
    print(
        '*** The charts and text reports are saved in "figs" and "texts" directories, respectively *** '
    )


if __name__ == "__main__":
    main()
