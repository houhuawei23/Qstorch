from numba import njit

import qstorch
import qstorch.fast_ops

# MAP
print("MAP")
tmap = qstorch.fast_ops.tensor_map(njit()(qstorch.operators.id))
out, a = qstorch.zeros((10,)), qstorch.zeros((10,))
tmap(*out.tuple(), *a.tuple())
print(tmap.parallel_diagnostics(level=3))

# ZIP
print("ZIP")
out, a, b = qstorch.zeros((10,)), qstorch.zeros((10,)), qstorch.zeros((10,))
tzip = qstorch.fast_ops.tensor_zip(njit()(qstorch.operators.eq))

tzip(*out.tuple(), *a.tuple(), *b.tuple())
print(tzip.parallel_diagnostics(level=3))

# REDUCE
print("REDUCE")
out, a = qstorch.zeros((1,)), qstorch.zeros((10,))
treduce = qstorch.fast_ops.tensor_reduce(njit()(qstorch.operators.add))

treduce(*out.tuple(), *a.tuple(), 0)
print(treduce.parallel_diagnostics(level=3))


# MM
print("MATRIX MULTIPLY")
out, a, b = (
    qstorch.zeros((1, 10, 10)),
    qstorch.zeros((1, 10, 20)),
    qstorch.zeros((1, 20, 10)),
)
tmm = qstorch.fast_ops.tensor_matrix_multiply

tmm(*out.tuple(), *a.tuple(), *b.tuple())
print(tmm.parallel_diagnostics(level=3))
