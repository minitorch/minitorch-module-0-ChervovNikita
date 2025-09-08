"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, List

# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


def mul(x: float, y: float) -> float:
    return x * y


def id(x: float) -> float:
    return x


def add(x: float, y: float) -> float:
    return x + y


def neg(x: float) -> float:
    return -x


def lt(x: float, y: float) -> bool:
    return x < y


def eq(x: float, y: float) -> bool:
    return x == y


def max(x: float, y: float) -> float:
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    return x if x >= 0 else 0


def log(x: float) -> float:
    return math.log(x)


def inv(x: float) -> float:
    return 1.0 / x


def log_back(x: float, y: float) -> float:
    return x / y


def inv_back(x: float, y: float) -> float:
    return (-1.0 / (x**2)) * y


def relu_back(x: float, y: float) -> float:
    df = 1 if x >= 0 else 0
    return df * y


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.


def map(f: Callable[[float], float], xs: List[float]) -> List[float]:
    return [f(x) for x in xs]


def zipWith(
    f: Callable[[float, float], float], xs: List[float], ys: List[float]
) -> List[float]:
    return [f(x, y) for x, y in zip(xs, ys)]


def reduce(f: Callable[[float, float], float], xs: List[float]) -> float:
    if not xs:
        return 0
    result = xs[0]
    for x in xs[1:]:
        result = f(result, x)
    return result


def negList(xs: List[float]) -> List[float]:
    return [neg(x) for x in xs]


def addLists(xs: List[float], ys: List[float]) -> List[float]:
    return [add(x, y) for x, y in zip(xs, ys)]


def sum(xs: List[float]) -> float:
    return reduce(add, xs)


def prod(xs: List[float]) -> float:
    return reduce(mul, xs)
