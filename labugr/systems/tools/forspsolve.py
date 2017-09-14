from __future__ import division, print_function, absolute_import

from warnings import warn

import numpy as np
from numpy import asarray, empty, ravel, nonzero
from .base import isspmatrix, SparseEfficiencyWarning
from .csc import isspmatrix_csc, csc_matrix
from .csr import isspmatrix_csr

from . import _superlu

useUmfpack = False

class MatrixRankWarning(UserWarning):
    pass

def spsolve(A, b, permc_spec=None, use_umfpack=True):
    """Solve the sparse linear system Ax=b, where b may be a vector or a matrix.

    Parameters
    ----------
    A : ndarray or sparse matrix
        The square matrix A will be converted into CSC or CSR form
    b : ndarray or sparse matrix
        The matrix or vector representing the right hand side of the equation.
        If a vector, b.shape must be (n,) or (n, 1).
    permc_spec : str, optional
        How to permute the columns of the matrix for sparsity preservation.
        (default: 'COLAMD')

        - ``NATURAL``: natural ordering.
        - ``MMD_ATA``: minimum degree ordering on the structure of A^T A.
        - ``MMD_AT_PLUS_A``: minimum degree ordering on the structure of A^T+A.
        - ``COLAMD``: approximate minimum degree column ordering
    use_umfpack : bool, optional
        if True (default) then use umfpack for the solution.  This is
        only referenced if b is a vector and ``scikit-umfpack`` is installed.

    Returns
    -------
    x : ndarray or sparse matrix
        the solution of the sparse linear equation.
        If b is a vector, then x is a vector of size A.shape[1]
        If b is a matrix, then x is a matrix of size (A.shape[1], b.shape[1])

    Notes
    -----
    For solving the matrix expression AX = B, this solver assumes the resulting
    matrix X is sparse, as is often the case for very sparse inputs.  If the
    resulting X is dense, the construction of this sparse result will be
    relatively expensive.  In that case, consider converting A to a dense
    matrix and using scipy.linalg.solve or its variants.

    Examples
    --------
    >>> from scipy.sparse import csc_matrix
    >>> from scipy.sparse.linalg import spsolve
    >>> A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)
    >>> B = csc_matrix([[2, 0], [-1, 0], [2, 0]], dtype=float)
    >>> x = spsolve(A, B)
    >>> np.allclose(A.dot(x).todense(), B.todense())
    True
    """
    if not (isspmatrix_csc(A) or isspmatrix_csr(A)):
        A = csc_matrix(A)
        warn('spsolve requires A be CSC or CSR matrix format',
                SparseEfficiencyWarning)

    # b is a vector only if b have shape (n,) or (n, 1)
    b_is_sparse = isspmatrix(b)
    if not b_is_sparse:
        b = asarray(b)
    b_is_vector = ((b.ndim == 1) or (b.ndim == 2 and b.shape[1] == 1))

    A.sort_indices()
    A = A.asfptype()  # upcast to a floating point format
    result_dtype = np.promote_types(A.dtype, b.dtype)
    if A.dtype != result_dtype:
        A = A.astype(result_dtype)
    if b.dtype != result_dtype:
        b = b.astype(result_dtype)

    # validate input shapes
    M, N = A.shape
    if (M != N):
        raise ValueError("matrix must be square (has shape %s)" % ((M, N),))

    if M != b.shape[0]:
        raise ValueError("matrix - rhs dimension mismatch (%s - %s)"
                         % (A.shape, b.shape[0]))

    use_umfpack = use_umfpack and useUmfpack

    if b_is_vector and use_umfpack:
        if b_is_sparse:
            b_vec = b.toarray()
        else:
            b_vec = b
        b_vec = asarray(b_vec, dtype=A.dtype).ravel()

        if noScikit:
            raise RuntimeError('Scikits.umfpack not installed.')

        if A.dtype.char not in 'dD':
            raise ValueError("convert matrix data to double, please, using"
                  " .astype(), or set linsolve.useUmfpack = False")

        umf = umfpack.UmfpackContext(_get_umf_family(A))
        x = umf.linsolve(umfpack.UMFPACK_A, A, b_vec,
                         autoTranspose=True)
    else:
        if b_is_vector and b_is_sparse:
            b = b.toarray()
            b_is_sparse = False

        if not b_is_sparse:
            if isspmatrix_csc(A):
                flag = 1  # CSC format
            else:
                flag = 0  # CSR format

            options = dict(ColPerm=permc_spec)
            x, info = _superlu.gssv(N, A.nnz, A.data, A.indices, A.indptr,
                                    b, flag, options=options)
            if info != 0:
                warn("Matrix is exactly singular", MatrixRankWarning)
                x.fill(np.nan)
            if b_is_vector:
                x = x.ravel()
        else:
            # b is sparse
            Afactsolve = factorized(A)

            if not isspmatrix_csc(b):
                warn('spsolve is more efficient when sparse b '
                     'is in the CSC matrix format', SparseEfficiencyWarning)
                b = csc_matrix(b)

            # Create a sparse output matrix by repeatedly applying
            # the sparse factorization to solve columns of b.
            data_segs = []
            row_segs = []
            col_segs = []
            for j in range(b.shape[1]):
                bj = b[:, j].A.ravel()
                xj = Afactsolve(bj)
                w = np.flatnonzero(xj)
                segment_length = w.shape[0]
                row_segs.append(w)
                col_segs.append(np.ones(segment_length, dtype=int)*j)
                data_segs.append(np.asarray(xj[w], dtype=A.dtype))
            sparse_data = np.concatenate(data_segs)
            sparse_row = np.concatenate(row_segs)
            sparse_col = np.concatenate(col_segs)
            x = A.__class__((sparse_data, (sparse_row, sparse_col)),
                           shape=b.shape, dtype=A.dtype)

    return x

def _get_umf_family(A):
    """Get umfpack family string given the sparse matrix dtype."""
    family = {'di': 'di', 'Di': 'zi', 'dl': 'dl', 'Dl': 'zl'}
    dt = A.dtype.char + A.indices.dtype.char
    return family[dt]
