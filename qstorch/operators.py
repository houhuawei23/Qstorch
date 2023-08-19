"""
Collection of the core mathematical operators used throughout the code base.
"""

from typing import Callable, Iterable
import numpy as np


def mul(x: float, y: float) -> float:
    """$f(x, y) = x * y$"""
    return x * y


def id(x: float) -> float:
    """$f(x) = x$"""
    return x


def add(x: float, y: float) -> float:
    """$f(x, y) = x + y$"""
    return x + y


def neg(x: float) -> float:
    """$f(x) = -x$"""
    return -x


def lt(x: float, y: float) -> float:
    """$f(x) =$ 1.0 if x is less than y else 0.0"""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """$f(x) =$ 1.0 if x is equal to y else 0.0"""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """$f(x) =$ x if x is greater than y else y"""
    return x if x > y else y


def is_close(x: float, y: float, epsilon: float = 1e-2) -> float:
    """$f(x) = |x - y| < 1e-2$"""
    return 1.0 if np.abs(x - y) < epsilon else 0.0


def sigmoid(x: float) -> float:
    r"""
    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$

    (See https://en.wikipedia.org/wiki/Sigmoid_function )

    Calculate as

    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$

    for stability.
    """
    return 1 / (1 + np.exp(-x)) if x >= 0 else np.exp(x) / (1 + np.exp(x))


def relu(x: float) -> float:
    """
    $f(x) =$ x if x is greater than 0, else 0

    (See https://en.wikipedia.org/wiki/Rectifier_(neural_networks) .)
    """
    return x if x > 0 else 0.0


def log(x: float, epsilon: float = 1e-6) -> float:
    """$f(x) = log(x)$"""
    return np.log(x + epsilon)


def exp(x: float) -> float:
    """$f(x) = e^{x}$"""
    return np.exp(x)


def log_back(x: float, d: float) -> float:
    r"""If $f = log$ as above, compute $d \times f'(x)$"""
    return d / x


def inv(x: float) -> float:
    """$f(x) = 1/x$"""
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    r"""If $f(x) = 1/x$ compute $d \times f'(x)$"""
    return -d / x ** 2


def relu_back(x: float, d: float) -> float:
    r"""If $f = relu$ compute $d \times f'(x)$"""
    return d if x >= 0.0 else 0.0

def sigmoid_back(x: float, d: float) -> float:
    r"""
    \frac{\mathrm{d} \text{sigmoid}}{\mathrm{d} x} =\text{sigmoid}(x)(1-\text{sigmoid}(x))
    """
    return d * sigmoid(x) * (1 - sigmoid(x))

def mul_back(x: float, y: float, d: float) -> float:
    r"""
    \frac{\partial xy}{\partial x} = y, \frac{\partial xy}{\partial y} = x
    """
    return d * y, d * x

def neg_back(d: float) -> float:
    return -1.0*d

def exp_back(a: float, d: float) -> float:
    return np.exp(a)*d

def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: Function from one value to one value.

    Returns:
         A function that takes a list, applies `fn` to each element, and returns a
         new list
    """

    def _map(x: Iterable[float]):
        return [fn(i) for i in x]

    return _map


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Use `map` and `neg` to negate each element in `ls`"""
    return map(neg)(ls)


def zipWith(
        fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """
    Higher-order zipwith (or map2).

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: combine two values

    Returns:
         Function that takes two equally sized lists `ls1` and `ls2`, produce a new list by
         applying fn(x, y) on each pair of elements.

    """

    def _zipWith(x: Iterable[float], y: Iterable[float]):
        return [fn(i, j) for i, j in zip(x, y)]

    return _zipWith


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add the elements of `ls1` and `ls2` using `zipWith` and `add`"""
    return zipWith(add)(ls1, ls2)


def reduce(
        fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""
    Higher-order reduce.

    Args:
        fn: combine two values
        start: start value $x_0$

    Returns:
         Function that takes a list `ls` of elements
         $x_1 \ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
         fn(x_1, x_0)))`
    """

    def _reduce(ls: Iterable[float]) -> float:
        result = start
        for x in ls:
            result = fn(result, x)
        return result

    return _reduce

def filter(fn: Callable[[float], bool]) -> Callable[[Iterable[float]], Iterable[float]]:
    def _filter(ls: Iterable[float]):
        ret = []
        for x in ls:
            if fn(x):
                ret.append(x)
        return ret

    return _filter

def sum(ls: Iterable[float]) -> float:
    """Sum up a list using `reduce` and `add`."""
    return reduce(add, 0)(ls)


def prod(ls: Iterable[float]) -> float:
    """Product of a list using `reduce` and `mul`."""
    return reduce(mul, 1)(ls)
