import numpy

from src.metrics import MetricTypes
from src.utils import DatasetMetadata, DatasetType

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
    categorical_columns=["sex", "smoker", "region"],
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
