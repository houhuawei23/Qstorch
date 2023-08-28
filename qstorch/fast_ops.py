from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(inline="always")(to_index)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        "See `tensor_ops.py`"

        # This line JIT compiles your tensor_map
        f = tensor_map(njit()(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        "See `tensor_ops.py`"

        f = tensor_zip(njit()(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        "See `tensor_ops.py`"
        f = tensor_reduce(njit()(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """
        Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
            a : tensor data a
            b : tensor data b

        Returns:
            New tensor data
        """

        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float]
) -> Callable[[Storage, Shape, Strides,
               Storage, Shape, Strides],
              None]:
    """
    NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
        fn: function mappings floats-to-floats to apply.

    Returns:
        Tensor map function.
    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = np.zeros(len(out_shape), dtype=np.int32)
        in_index = np.zeros(len(in_shape), dtype=np.int32)
        for i in range(out.size):
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            data = in_storage[index_to_position(in_index, in_strides)]
            out[index_to_position(out_index, out_strides)] = fn(data)

    return njit(parallel=True)(_map)


def tensor_zip(
    fn: Callable[[float, float], float]
) -> Callable[[Storage, Shape, Strides,
               Storage, Shape, Strides,
               Storage, Shape, Strides],
              None
              ]:
    """
    NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.


    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
        fn: function maps two floats to float to apply.

    Returns:
        Tensor zip function.
    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = np.zeros(len(out_shape), dtype=np.int32)
        a_index = np.zeros(len(a_shape), dtype=np.int32)
        b_index = np.zeros(len(b_shape), dtype=np.int32)
        for i in prange(out.size):
            to_index(i, out_shape, out_index)
            op = index_to_position(out_index, out_strides)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            ap = index_to_position(a_index, a_strides)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            bp = index_to_position(b_index, b_strides)
            out[op] = fn(a_storage[ap], b_storage[bp])

    return njit(parallel=True)(_zip)


def tensor_reduce(
    fn: Callable[[float, float], float]
) -> Callable[[Storage, Shape, Strides,
               Storage, Shape, Strides,
               int],
              None]:
    """
    NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
        fn: reduction function mapping two floats to float.

    Returns:
        Tensor reduce function
    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        n = len(out)
        out_dims = len(out_shape)
        for i in prange(n):
            out_index = np.zeros(out_dims, dtype=np.int32)
            to_index(i,out_shape,out_index)
            out_idx = index_to_position(out_index,out_strides)

            reduce_dim_size = a_shape[reduce_dim]

            for j in range(reduce_dim_size):
                idx_a = out_index.copy()
                idx_a[reduce_dim] = j
                pos_a = index_to_position(idx_a, a_strides)
                out[out_idx] = fn(out[out_idx],a_storage[pos_a])
        # out_index = np.zeros(len(out_shape), dtype=np.int32)

        # for i in prange(out.size):
        #     to_index(i, out_shape, out_index)
        #     pos = index_to_position(out_index, out_strides)
        #     for j in range(a_shape[reduce_dim]):
        #         a_index = out_index.copy()
        #         a_index[reduce_dim] = j
        #         apos = index_to_position(a_index, a_strides)
        #         out[pos] = fn(a_storage[apos], out[pos])

    return njit(parallel=True)(_reduce)  # type: ignore


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """
    NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """
    # check dims match
    assert a_shape[-1] == b_shape[-2]

    # a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    # b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    
    n = out.size

    for i in prange(n):
        # get indexes
        out_index = out_shape.copy()
        ti = i + 0
        for ki in range(len(out_shape) - 1, -1, -1):
            out_index[ki] = ti % out_shape[ki]
            ti = ti // out_shape[ki]
        out_pos = 0
        for k, v in enumerate(out_index):
            out_pos += v * out_strides[k]
        last_dim = a_shape[-1]
        for j in range(last_dim):
            tj = j + 0
            a_index = a_shape.copy()
            a_loop_index = out_index.copy()
            a_loop_index[-1] = tj
            # broadcast_index(a_loop_index, out_shape, a_shape, a_index)
            # def broadcast_index(
            #     big_index: Index,
            #     big_shape: Shape,
            #     shape: Shape,
            #     out_index: OutIndex
            # )
            for ki in range(a_shape.size):
                offset = ki + out_shape.size - a_shape.size
                a_index[ki] = a_loop_index[offset] if a_shape[ki] != 1 else 0

            a_pos = 0
            for k, v in enumerate(a_index):
                a_pos += v * a_strides[k]
            
            b_index = b_shape.copy()
            b_loop_index = out_index.copy()
            b_loop_index[-2] = tj
            # broadcast_index(b_loop_index, out_shape, b_shape, b_index)
            for ki in range(b_shape.size):
                offset = ki + out_shape.size - b_shape.size
                b_index[ki] = b_loop_index[offset] if b_shape[ki] != 1 else 0

            b_pos = 0
            for k, v in enumerate(b_index):
                b_pos += v * b_strides[k]
            out[out_pos] += a_storage[a_pos] * b_storage[b_pos]

tensor_matrix_multiply = njit(parallel=True, fastmath=True)(_tensor_matrix_multiply)

# do not use the function below
def batched_mm(a: Tensor, b: Tensor) -> Tensor:
    """
    Batched matrix multiplication.
    """
    if a.shape[-1] != b.shape[-2]:
        raise IndexError("a and b cannot multiply!")
    a_2dflag = False
    if len(a.shape) == 2:
        a = a.contiguous().view(1, a.shape[0], a.shape[1])
        a_2dflag = True
    b_2dflag = False
    if len(b.shape) == 2:
        b = b.contiguous().view(1, b.shape[0], b.shape[1])
        b_2dflag = True
    both2d = a_2dflag and b_2dflag
    out_shape = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
    out_shape.append(a.shape[-2])
    out_shape.append(b.shape[-1])

    out = a.zeros(tuple(out_shape))
    tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())
    if both2d:
        out = out.view(out.shape[1], out.shape[2])
    return out
