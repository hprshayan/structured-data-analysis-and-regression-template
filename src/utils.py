from dataclasses import dataclass
from functools import reduce
import os
import shutil
import matplotlib.pyplot as plt
from typing import Callable, Literal
import numpy
import pandas
from sklearn.model_selection import train_test_split, permutation_test_score


@dataclass
class DatasetMetadata:
    path: str
    target: list[str]
    header: int = 0  # the row index that includes the headers
    skip_columns: int = 0  # number of skipped columns


def double_dim_converter(x: numpy.ndarray) -> numpy.ndarray:
    return x.reshape(-1, 1) if x.ndim == 1 else x


def load_dataset(path: str, header: int = 0, column_skip_count: int = 0):
    with open(path, "rb") as afile:
        data_frame = pandas.read_excel(afile, header=header)
    return data_frame.drop(data_frame.columns[:column_skip_count], axis=1)


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
    save_train_test_data: bool = True,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:

    X, y = separate_feature_target(data_frame, target)
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

    return split


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
