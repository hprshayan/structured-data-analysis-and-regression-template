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
from src.metrics import Metrics, calculate_err_metrics
from src.utils import (
    DatasetMetadata,
    create_directories,
    data_train_test_split,
    write_to_file,
)
from src.model import GridSearchScenario, ModelType, Pipeline, STDScaler


warnings.filterwarnings("ignore", category=DataConversionWarning)

# dataset_metadata = DatasetMetadata("1.xlsx", ["Vs"], 0, 0)
dataset_metadata = DatasetMetadata("2.xlsx", ["Nw[0-2]", "No[2-6]"], 1, 1)

TEST_SIZE = 0.2
SEED = 0
RANDOM_FORREST_HPARAMS = {
    "max_depth": list(range(1, 20)),
    "n_estimators": list(range(20, 200, 20)),
}
RIDGE_HPARAMS = {"alpha": list(numpy.arange(0.1, 2.0, 0.1))}
LASSO_HPARAMS = {"alpha": list(numpy.arange(0.1, 2.0, 0.1))}
MLP_HPARAMS = {
    "hidden_layer_sizes": [(10,), (20,), (100,), (10, 10), (20, 20)],
    "activation": ["identity", "relu"],
    "batch_size": [1, 20, 50, 100],
    "learning_rate": ["constant", "invscaling", "adaptive"],
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
    path: str = "texts/model_metrics.txt",
) -> None:
    def execute(scenario: GridSearchScenario) -> Metrics:
        print(
            f"performing hparam grid search for the {scenario.model_type.value} model..."
        )
        pipeline = Pipeline(scenario.model, scenario.model_type)
        pipeline.fit(train_features, train_targets, hparams=scenario.hparams)
        return calculate_err_metrics(test_targets, pipeline.forward(test_features))

    execution_scenarios = [
        GridSearchScenario(ModelType.LINEAR, LinearRegression, {}),
        GridSearchScenario(
            ModelType.RANDOM_FORREST, RandomForestRegressor, RANDOM_FORREST_HPARAMS
        ),
        GridSearchScenario(ModelType.RIDGE, Ridge, RIDGE_HPARAMS),
        GridSearchScenario(ModelType.LASSO, Lasso, LASSO_HPARAMS),
        GridSearchScenario(ModelType.MLP, MLPRegressor, MLP_HPARAMS),
    ]

    model_metrics = {
        scenario.model_type.value: execute(scenario) for scenario in execution_scenarios
    }

    write_to_file(pformat(model_metrics), path=path)


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
    execute_hparam_search(train_features, test_features, train_targets, test_targets)

    print("\nall done!")
    print(
        '*** The charts and text reports are saved in "figs" and "texts" directories, respectively *** '
    )


if __name__ == "__main__":
    main()
