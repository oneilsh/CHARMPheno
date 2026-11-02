"""Row-sparse sufficient statistics: a (n_rows, V) matrix that stores only its touched rows.

WHY. A model whose E-step restricts each record to a subset of the rows of a (K, V)
statistic (e.g. a gated topic model, where a document may update only its allowed
topics) leaves most rows of a partition's partial at exactly zero. Allocating, filling
and shipping the dense (K, V) partial anyway costs K * V * 8 bytes per partition
(hundreds of MB at K in the thousands and V in the ten-thousands), executor heap and
executor -> driver traffic alike. Storing the touched rows only, each full-V wide,
costs n_touched * V * 8.

Rows, not cells: when every record also touches a common block of rows (a shared
background block, say) over all of its columns, a (row, column) coordinate list would
store that block once per record and grow past the dense size; whole rows store it
once per partition.

EXACTNESS. Every operation here reproduces the dense arithmetic bit for bit: an
untouched row is exactly zero, and every combination either copies a value (where
the dense path computes x + 0.0 == x) or adds the same two floats the dense path adds.
Scaling by a scalar is elementwise on the stored rows (0 * s == 0 elsewhere).

Plain numpy, no scipy.sparse: scipy's `*` on a sparse matrix is a matrix product, and
the runners scale every statistic with `v * scale`, so the type defines `*` itself
(scalars only) and refuses numpy ufuncs (`__array_ufunc__ = None`), which makes
`ndarray * RowSparseStats` and `np.add(...)` dispatch here or fail loudly instead of
silently densifying. `np.asarray(x)` still densifies (via `__array__`) for readers that
want the plain array; arithmetic code should call `as_dense` explicitly.
"""
from __future__ import annotations

import numbers

import numpy as np

# combine: a row union covering more than this fraction of the rows is returned dense.
# Past ~half the rows the compact form saves under 2x and the union would only grow
# further up the reduce tree, so the upper levels (and the driver) hold one plain
# array instead of a near-full compact copy plus its row index.
DENSIFY_FRACTION = 0.5


class RowSparseStats:
    """A (n_rows, V) float64 matrix stored as its touched rows.

    `rows`: sorted, unique int64 row ids; `values`: (len(rows), V) float64, row i of
    which is matrix row rows[i]. Every other row is exactly zero."""

    __slots__ = ("rows", "values", "n_rows")
    __array_ufunc__ = None          # numpy defers binary operators to this class

    def __init__(self, rows, values, n_rows: int) -> None:
        rows = np.asarray(rows, dtype=np.int64)
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != rows.shape[0]:
            raise ValueError(f"values must be (len(rows), V); got {values.shape} "
                             f"for {rows.shape[0]} rows")
        if rows.size and (np.any(np.diff(rows) <= 0) or rows[0] < 0
                          or rows[-1] >= n_rows):
            raise ValueError("rows must be sorted, unique and within [0, n_rows)")
        self.rows = rows
        self.values = values
        self.n_rows = int(n_rows)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_rows, self.values.shape[1])

    @property
    def nbytes(self) -> int:
        return int(self.rows.nbytes + self.values.nbytes)

    def toarray(self) -> np.ndarray:
        out = np.zeros(self.shape, dtype=np.float64)
        out[self.rows] = self.values
        return out

    def __array__(self, dtype=None, copy=None):
        out = self.toarray()
        return out if dtype is None else out.astype(dtype, copy=False)

    def __mul__(self, s):
        if not (isinstance(s, numbers.Number) or np.ndim(s) == 0):
            return NotImplemented   # scalar scaling only; never a matrix product
        return RowSparseStats(self.rows, self.values * s, self.n_rows)

    __rmul__ = __mul__

    def __reduce__(self):
        return (RowSparseStats, (self.rows, self.values, self.n_rows))

    def __repr__(self) -> str:
        return (f"RowSparseStats(shape={self.shape}, touched_rows={self.rows.size}, "
                f"nbytes={self.nbytes})")


def as_dense(x) -> np.ndarray:
    """The plain ndarray behind a statistic: densify a RowSparseStats, pass an array
    (or anything array-like) through `np.asarray`."""
    if isinstance(x, RowSparseStats):
        return x.toarray()
    return np.asarray(x)


def add_row_sparse(a, b, *, densify_fraction: float = DENSIFY_FRACTION):
    """a + b for statistics that are each a RowSparseStats or a dense ndarray.

    Dense with anything -> dense (a fresh array; inputs are not mutated). Two
    row-sparse -> the row union, returned dense once it covers more than
    `densify_fraction` of the rows. Commutative, and equal value-for-value to the
    dense sum of the densified inputs (so a reduce tree gives the dense path's bits)."""
    a_sp, b_sp = isinstance(a, RowSparseStats), isinstance(b, RowSparseStats)
    if not a_sp and not b_sp:
        return np.asarray(a) + np.asarray(b)
    if a_sp and b_sp:
        if a.shape != b.shape:
            raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
        union = np.union1d(a.rows, b.rows)
        if union.size > densify_fraction * a.n_rows:
            out = np.zeros(a.shape, dtype=np.float64)
            out[a.rows] = a.values
            out[b.rows] += b.values
            return out
        out = np.zeros((union.size, a.values.shape[1]), dtype=np.float64)
        out[np.searchsorted(union, a.rows)] = a.values
        out[np.searchsorted(union, b.rows)] += b.values
        return RowSparseStats(union, out, a.n_rows)
    dense, sp = (np.asarray(b), a) if a_sp else (np.asarray(a), b)
    if dense.shape != sp.shape:
        raise ValueError(f"shape mismatch: {dense.shape} vs {sp.shape}")
    out = np.array(dense, dtype=np.float64, copy=True)
    out[sp.rows] += sp.values
    return out
