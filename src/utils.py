import os
import shutil
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import reduce
from typing import Callable, Literal, Protocol

import matplotlib.pyplot as plt
import numpy
import pandas
from sklearn.model_selection import permutation_test_score, train_test_split


class DatasetType(Enum):
    CSV = auto()
    XLSX = auto()


@dataclass
class DatasetMetadata:
    path: str
    data_format: DatasetType  # the format of dataset e,g,: csv, xlsx
    target: list[str]  # the list of target columns
    header: int = 0  # the row index that includes the headers
    skip_columns: int = 0  # number of skipped columns
    delimiter: str | None = None  # The delimiter of csv files
    na_values: list[str] = field(
        default_factory=list
    )  # list of null values in the original dataset
    categorical_columns: list[str] = field(default_factory=list)


class CategoricalEncoderDecoder:
    def __init__(
        self, dataset_metadata: DatasetMetadata, all_columns: list[str]
    ) -> None:
        self._prefix_sep: str = "_"
        self._categorical_columns: list[str] = dataset_metadata.categorical_columns
        self._target: list[str] = dataset_metadata.target
        tmp_columns = self._target + self._categorical_columns
        self._numerical_columns: list[str] = [
            col for col in all_columns if col not in tmp_columns
        ]
        self._decode_columns: list[str] | None = None
        self._col_name_idx_dict: dict[str, int] | None = None

    def encode(self, data: pandas.DataFrame) -> pandas.DataFrame:
        if len(self._categorical_columns) == 0:
            return data
        encoded = pandas.get_dummies(
            data[self._categorical_columns], prefix_sep=self._prefix_sep
        )
        self._decode_columns = encoded.columns
        return pandas.concat(
            [data[self._numerical_columns], encoded, data[self._target]], axis=1
        )

    def encode_features(self, data: numpy.ndarray) -> numpy.ndarray:
        if len(self._categorical_columns) == 0:
            return data
        categorical_indices = [
            self._col_name_idx_dict[col] for col in self._categorical_columns
        ]
        numerical_indices = [
            self._col_name_idx_dict[col] for col in self._numerical_columns
        ]
        encoded = pandas.get_dummies(
            pandas.DataFrame(data[:, categorical_indices]), prefix_sep=self._prefix_sep
        ).to_numpy()
        return numpy.concatenate((data[:, numerical_indices], encoded), axis=1)

    def decode(self, data: pandas.DataFrame) -> pandas.DataFrame:
        if len(self._categorical_columns) == 0:
            return data
        return pandas.concat(
            [
                data[self._numerical_columns],
                pandas.from_dummies(data[self._decode_columns], sep=self._prefix_sep),
                data[self._target],
            ],
            axis=1,
        )

    def set_name_idx(self, name_idx_dict: dict[str, int]) -> None:
        self._col_name_idx_dict = name_idx_dict


class DataLoader(Protocol):
    "represents the class that loads dataset"

    @staticmethod
    def load(dataset_metadata: DatasetMetadata) -> pandas.DataFrame:
        ...


class XlsxLoader:
    @staticmethod
    def load(dataset_metadata: DatasetMetadata) -> pandas.DataFrame:
        with open(dataset_metadata.path, "rb") as afile:
            data_frame = pandas.read_excel(afile, header=dataset_metadata.header)

        return data_frame.drop(
            data_frame.columns[: dataset_metadata.skip_columns], axis=1
        )


class CsvLoader:
    @staticmethod
    def load(dataset_metadata: DatasetMetadata) -> pandas.DataFrame:
        with open(dataset_metadata.path, "r") as afile:
            data_frame = pandas.read_csv(
                afile,
                delimiter=dataset_metadata.delimiter,
                header=dataset_metadata.header,
                na_values=dataset_metadata.na_values,
            )
        return data_frame.drop(
            data_frame.columns[: dataset_metadata.skip_columns], axis=1
        )


def load_dataset(dataset_metadata: DatasetMetadata) -> pandas.DataFrame:
    if dataset_metadata.data_format == DatasetType.CSV:
        return CsvLoader().load(dataset_metadata)
    elif dataset_metadata.data_format == DatasetType.XLSX:
        return XlsxLoader.load(dataset_metadata)


def drop_save_null_rows(data_frame: pandas.DataFrame) -> pandas.DataFrame:
    rows_with_nulls = data_frame.isnull().any(axis=1)
    with pandas.ExcelWriter("rows-with-nulls.xlsx") as writer:
        data_frame[rows_with_nulls].to_excel(
            writer, sheet_name="dropped-datapoints", index=True
        )
    return data_frame[~rows_with_nulls]


def encode_categorical_data(
    dataset: pandas.DataFrame, dataset_metadata: DatasetMetadata, prefix_sep: str
) -> pandas.DataFrame:
    new_column_order = [
        col
        for col in dataset.columns
        if col not in dataset_metadata.categorical_columns
    ] + dataset_metadata.categorical_columns
    return pandas.get_dummies(dataset[new_column_order], prefix_sep=prefix_sep)


def load_and_init_process_data(dataset_metadata: DatasetMetadata) -> pandas.DataFrame:
    original_dataset = load_dataset(dataset_metadata)
    dataset_without_null = drop_save_null_rows(original_dataset)
    return dataset_without_null


def double_dim_converter(x: numpy.ndarray) -> numpy.ndarray:
    return x.reshape(-1, 1) if x.ndim == 1 else x


# credicts to https://www.geeksforgeeks.org/function-composition-in-python/
def composite_function(
    *func: tuple[Callable[[numpy.ndarray], numpy.ndarray]]
) -> Callable[[numpy.ndarray], numpy.ndarray]:
    def compose(f, g):
        return lambda x: f(g(x))

    return reduce(compose, func, lambda x: x)


def separate_feature_target(
    data_frame: pandas.DataFrame, target: list[str]
) -> tuple[numpy.ndarray, numpy.ndarray, dict[str, int]]:
    dropped_target = data_frame.drop(target, axis=1)
    X = double_dim_converter(dropped_target.to_numpy())
    y = double_dim_converter(data_frame[target].to_numpy())
    col_name_idx: dict[str, int] = {
        name: idx for idx, name in enumerate(dropped_target.columns)
    }
    return X, y, col_name_idx


def data_train_test_split(
    data_frame: pandas.DataFrame,
    test_size: float,
    seed: int,
    target: list[str],
    save_train_test_data: bool = True,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, dict[str, int]]:
    X, y, col_name_idx = separate_feature_target(data_frame, target)
    split = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=seed)

    if save_train_test_data:
        with pandas.ExcelWriter("train-test-dataset.xlsx") as writer:
            train_features, test_features, train_targets, test_targets = split
            train_data_frame = pandas.DataFrame(
                numpy.concatenate((train_features, train_targets), axis=1),
                columns=data_frame.columns,
            )
            test_data_frame = pandas.DataFrame(
                numpy.concatenate((test_features, test_targets), axis=1),
                columns=data_frame.columns,
            )

            train_data_frame.to_excel(writer, sheet_name="training-data", index=False)
            test_data_frame.to_excel(writer, sheet_name="test-data", index=False)

    return *split, col_name_idx


def write_to_file(*text: tuple[str], path: str, mode: Literal["w", "a"] = "w"):
    with open(path, mode, encoding="utf-8") as afile:
        print(*text, sep="\n", file=afile)


def export_permutation_test_score(
    model: any,
    X: numpy.ndarray,
    y: numpy.ndarray,
    scoring: str = "r2",
    dpi: int = 200,
    path: str = "figs/best_model_permutation_test_score.png",
) -> None:
    score, perm_scores, pvalue = permutation_test_score(
        model, X, y, scoring=scoring, n_permutations=100
    )

    fig, ax = plt.subplots()
    ax.hist(perm_scores, bins=20, density=True)
    ax.axvline(score, ls="--", color="r")
    score_label = f"Score on original\ndata: {score:.2f}\n(p-value: {pvalue:.3f})"
    ax.text(0.7, 10, score_label, fontsize=12)
    ax.set_xlabel(f"{scoring} score")
    _ = ax.set_ylabel("Probability")

    plt.savefig(path, dpi=dpi)


def create_directories() -> None:
    dirs = ["figs", "texts"]
    _ = [shutil.rmtree(dir, ignore_errors=True) for dir in dirs]
    _ = [os.mkdir(dir) for dir in dirs]
