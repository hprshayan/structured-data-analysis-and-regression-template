"Analyzing a dataset and fitting regression models on it"

import warnings
import numpy
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from pprint import pprint
from sklearn.neural_network import MLPRegressor

from sklearn.svm import SVR

from src.data_analysis import DatasetAnalysis
from src.metrics import Metrics, calculate_err_metrics
from src.utils import data_train_test_split
from src.model import GridSearchScenario, ModelType, Pipeline, STDScaler


warnings.filterwarnings("ignore", category=DataConversionWarning)

DS_PATH = "1.xlsx"
TARGET = ["Vs"]
HEADER = 0
SKIP_COLUMNS = 0

# DS_PATH = "2.xlsx"
# TARGET = ["Nw[0-2]", "No[2-6]"]
# HEADER = 1
# SKIP_COLUMNS = 1

TEST_SIZE = 0.2
SEED = 0
RANDOM_FORREST_HPARAMS = {"max_depth": list(range(1, 20)), "n_estimators": list(range(20, 200, 20))}
RIDGE_HPARAMS = {"alpha": list(numpy.arange(0.1, 2.0, 0.1))}
LASSO_HPARAMS = {"alpha": list(numpy.arange(0.1, 2.0, 0.1))}
SVR_HPARAMS = {"C": list(numpy.arange(0.1, 2.0, 0.2)), "epsilon": list(numpy.arange(0.05, 0.2, 0.01))}
MLP_HPARAMS = {"hidden_layer_sizes": [(10,), (20,), (100,), (10, 10), (20, 20)],
               "activation": ["identity", "relu"],
               "batch_size": [1, 20, 50, 100],
               "learning_rate": ["constant", "invscaling", "adaptive"]}


def load_dataset(path: str, header: int = 0, column_skip_count: int = 0):
    with open(path, "rb") as afile:
        data_frame = pd.read_excel(afile, header=header)
    return data_frame.drop(data_frame.columns[:column_skip_count], axis=1)


def execute_hparam_search(
    train_features: numpy.ndarray,
    test_features: numpy.ndarray,
    train_targets: numpy.ndarray,
    test_targets: numpy.ndarray,
    path: str = "texts/model_metrics.txt"
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
        GridSearchScenario(ModelType.SVR, SVR, SVR_HPARAMS),
        GridSearchScenario(ModelType.MLP, MLPRegressor, MLP_HPARAMS),
    ]
    
    model_metrics = {scenario.model_type.value: execute(scenario) for scenario in execution_scenarios}

    with open(path, "w", encoding="utf-8") as afile: 
        pprint(model_metrics, stream=afile)


def main():
    dataset = load_dataset(DS_PATH, header=HEADER, column_skip_count=SKIP_COLUMNS)

    # Analyze the dataset
    print("analyzing the dataset...")
    dataset_analyzer = DatasetAnalysis(dataset, TARGET)
    dataset_analyzer.analyze_dataset()

    train_features, test_features, train_targets, test_targets = data_train_test_split(
        dataset, TEST_SIZE, SEED, TARGET
    )

    # fitting the dataset transformers
    feature_scaler = STDScaler()
    target_scaler = STDScaler()
    feature_scaler.fit(train_features)
    target_scaler.fit(train_targets)
    Pipeline.feature_scaler = feature_scaler
    Pipeline.target_scaler = target_scaler

    # run the hyperparameter search
    # print("\nexecuting hyperparameter tunning for regression models...")
    # model_metrics = execute_hparam_search(
    #     train_features, test_features, train_targets, test_targets
    # )
    # print("all done!")


if __name__ == "__main__":
    main()
