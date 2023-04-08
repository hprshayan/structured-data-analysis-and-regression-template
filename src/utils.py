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


def export_permutation_test_score(
    model: any,
    X: numpy.ndarray,
    y: numpy.ndarray,
    scoring: str = "r2",
    dpi: int = 200,
    path: str = "figs/best_model_permutation_test_score.png",
) -> None:
    score, perm_scores, pvalue = permutation_test_score(
        model, X, y, scoring=scoring, n_permutations=1000
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
