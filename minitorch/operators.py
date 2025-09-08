"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

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

def mul(x, y):
    return x * y


def id(x):
    return x


def add(x, y):
    return x + y


def neg(x):
    return -x
    

def lt(x, y):
    return x < y


def eq(x, y):
    return x == y


def max(x, y):
    return x if x > y else y


def is_close(x, y):
    return abs(x - y) < 1e-2


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    return math.exp(x) / (1.0 + math.exp(x))


def relu(x):
    return x if x >= 0 else 0


def log(x):
    return math.log(x)


def inv(x):
    return 1.0 / x


def log_back(x, y):
    return x / y


def inv_back(x, y):
    return (-1.0 / (x ** 2)) * y


def relu_back(x, y):
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


def map(f, xs):
    return [f(x) for x in xs]


def zipWith(f, xs, ys):
    return [f(x, y) for x, y in zip(xs, ys)]


def reduce(f, xs):
    if not xs:
        return 0
    result = xs[0]
    for x in xs[1:]:
        result = f(result, x)
    return result


def negList(xs):
    return [neg(x) for x in xs]


def addLists(xs, ys):
    return [add(x, y) for x, y in zip(xs, ys)]


def sum(xs):
    return reduce(add, xs)


def prod(xs):
    return reduce(mul, xs)
