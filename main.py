"Analyzing a dataset and fitting regression models on it"

import warnings
import numpy
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from pprint import pformat
from sklearn.neural_network import MLPRegressor

from src.data_analysis import DatasetAnalysis
from src.metrics import MetricTypes, Metrics, calculate_err_metrics
from src.utils import (
    DatasetMetadata,
    create_directories,
    data_train_test_split,
    export_permutation_test_score,
    write_to_file,
)
from src.model import GridSearchScenario, ModelType, Pipeline, STDScaler


warnings.filterwarnings("ignore", category=DataConversionWarning)

dataset_metadata = DatasetMetadata("1.xlsx", ["Vs"], 0, 0)
# dataset_metadata = DatasetMetadata("2.xlsx", ["Nw[0-2]", "No[2-6]"], 1, 1)

COMPARISON_CRITERIA = MetricTypes.MSE
TEST_SIZE = 0.2
SEED = 0
RANDOM_FORREST_HPARAMS = {
    "max_depth": list(range(1, 20)),
    "n_estimators": list(range(20, 200, 20)),
}
RIDGE_HPARAMS = {"alpha": list(numpy.arange(0.1, 2.0, 0.1))}
LASSO_HPARAMS = {"alpha": list(numpy.arange(0.1, 2.0, 0.1))}
MLP_HPARAMS = {
    "hidden_layer_sizes": [(10,), (20,), (100,), (10, 10)],
    "activation": ["identity", "relu"],
    "batch_size": [1, 20, 50],
    "learning_rate": ["constant", "adaptive"],
    "learning_rate_init": [0.001, 0.01],
}


def load_dataset(path: str, header: int = 0, column_skip_count: int = 0):
    with open(path, "rb") as afile:
        data_frame = pd.read_excel(afile, header=header)
    return data_frame.drop(data_frame.columns[:column_skip_count], axis=1)


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
        if (
            getattr(model_metrics[scenario.model_type.value], comparison_criteria.value)
            < best_metric
        ):
            best_pipeline = pipeline

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

    # load the dataset
    dataset = load_dataset(
        dataset_metadata.path,
        header=dataset_metadata.header,
        column_skip_count=dataset_metadata.skip_columns,
    )

    # Analyze the dataset
    print("analyzing the dataset...")
    dataset_analyzer = DatasetAnalysis(dataset, dataset_metadata.target)
    dataset_analyzer.analyze_dataset()

    train_features, test_features, train_targets, test_targets = data_train_test_split(
        dataset, TEST_SIZE, SEED, dataset_metadata.target
    )

    # fitting the dataset transformers
    feature_scaler = STDScaler()
    target_scaler = STDScaler()
    feature_scaler.fit(train_features)
    target_scaler.fit(train_targets)
    Pipeline.feature_scaler = feature_scaler
    Pipeline.target_scaler = target_scaler

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
