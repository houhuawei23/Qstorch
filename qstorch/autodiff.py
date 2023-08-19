from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple
import numpy as np
from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-5) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    if not callable(f):
        raise TypeError("The provided 'f' argument must be a callable function.")

    if arg >= len(vals):
        raise ValueError("The provided 'arg' index is out of range.")
    vals = list(vals)
    vals[arg] = vals[arg] + epsilon/2
    h1 = f(*vals)
    vals[arg] = vals[arg] - epsilon
    h2 = f(*vals)
    return (h1 - h2) / epsilon


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    visited: [int] = []
    queue: [Variable] = []
    def dfs(v: Variable) -> None:
        if v.is_constant():
            return
        if v.unique_id in visited:
            return
        if not v.is_leaf():
            for p in v.parents:
                dfs(p)
        visited.append(v.unique_id)
        queue.insert(0, v)
    dfs(variable)
    return queue

def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    topo = topological_sort(variable)
    nodes_deriv = {variable.unique_id: deriv}
    for v in topo:
        if v.is_leaf():
            continue
        if v.unique_id in nodes_deriv.keys():
            deriv = nodes_deriv[v.unique_id]
        back_deriv = v.chain_rule(deriv)
        for inputs, grad in back_deriv:
            if inputs.is_leaf():
                inputs.accumulate_derivative(grad)
                continue
            if inputs.unique_id not in nodes_deriv.keys():
                nodes_deriv[inputs.unique_id] = grad
            else:
                nodes_deriv[inputs.unique_id] += grad




@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
