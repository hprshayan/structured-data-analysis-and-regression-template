from dataclasses import dataclass
from functools import reduce
import os
import shutil
from typing import Callable, Literal
import numpy
import pandas
from sklearn.model_selection import train_test_split


@dataclass
class DatasetMetadata:
    path: str
    target: list[str]
    header: int = 0  # the row index that includes the headers
    skip_columns: int = 0  # number of skipped columns


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
    data_frame: pandas.core.frame.DataFrame, target: list[str]
) -> tuple[numpy.ndarray, numpy.ndarray]:
    X = double_dim_converter(data_frame.drop(target, axis=1).to_numpy())
    y = double_dim_converter(data_frame[target].to_numpy())
    return X, y


def data_train_test_split(
    data_frame: pandas.core.frame.DataFrame,
    test_size: float,
    seed: int,
    target: list[str],
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    X, y = separate_feature_target(data_frame, target)
    return train_test_split(X, y, test_size=test_size, shuffle=True, random_state=seed)


def write_to_file(*text: tuple[str], path: str, mode: Literal["w", "a"] = "w"):
    with open(path, mode, encoding="utf-8") as afile:
        print(*text, sep="\n", file=afile)


def create_directories() -> None:
    dirs = ["figs", "texts"]
    _ = [shutil.rmtree(dir, ignore_errors=True) for dir in dirs]
    _ = [os.mkdir(dir) for dir in dirs]
