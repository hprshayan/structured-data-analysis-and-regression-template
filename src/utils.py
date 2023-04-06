from functools import reduce
from typing import Callable
import numpy
import pandas
from sklearn.model_selection import train_test_split

def double_dim_converter(x: numpy.ndarray) -> numpy.ndarray:
    return x.reshape(-1, 1) if x.ndim == 1 else x

# credicts to https://www.geeksforgeeks.org/function-composition-in-python/
def composite_function(*func: tuple[Callable]) -> Callable:

    def compose(f, g):
        return lambda x: f(g(x))
    
    return reduce(compose, func, lambda x: x)

def separate_feature_target(data_frame: pandas.core.frame.DataFrame, target: list[str]) -> tuple[numpy.ndarray, numpy.ndarray]:
    X = double_dim_converter(data_frame.drop(target, axis=1).to_numpy())
    y = double_dim_converter(data_frame[target].to_numpy())
    return X, y

def data_train_test_split(data_frame: pandas.core.frame.DataFrame, test_size: float, seed: int, target: list[str]) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    X, y = separate_feature_target(data_frame, target)
    return train_test_split(X, y, test_size=test_size, shuffle=True, random_state=seed)
