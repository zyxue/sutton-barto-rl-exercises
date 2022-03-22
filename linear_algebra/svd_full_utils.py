# TODO(zhuyi): This code is NOT working yet, prefer svd_reduced_utils.py
#
# It turns out full SVD is trickier to program than reduced SVD because of
# needed padding of the singular value matrix to match the shape of U, and
# corresponding padding and sorting of eigenvectors in U and V.

from optparse import Optional
from typing import List, Optional, Tuple

import numpy as np
import sympy as sp


def eigen_decompose(A, sort_eigen_vals=True) -> Tuple[List, List]:
    """Eigen decompose A.

    Args:
        sort_eigen_vals: whether to sort eigenvalues in reversed order or not.

    Returns:
        A tuple of (list of eigenvalues, list of eigenvectors).
    """
    vals, vects = [], []  # eigenvalues and eigenvectors

    decomposed = A.eigenvects()
    if sort_eigen_vals:
        decomposed = sorted(
            decomposed,
            key=lambda v: v[0],
            reverse=True,
        )

    for lambda_, multiplicity, xs in decomposed:
        for i in range(multiplicity):
            vals.append(lambda_)
            vects.append(xs[i])

    return vals, vects


def pad_or_trim_Σ(D: sp.Matrix, A: sp.Matrix) -> sp.Matrix:
    m, n = A.shape

    if D.shape[0] < m:
        D = sp.Matrix(
            np.vstack(
                [
                    D,
                    np.zeros(
                        shape=(
                            m - D.shape[0],
                            D.shape[1],
                        ),
                    ).reshape(1, -1),
                ]
            )
        )

    if D.shape[1] < n:
        D = sp.Matrix(
            np.hstack(
                [
                    D,
                    np.zeros(
                        shape=(
                            D.shape[0],
                            n - D.shape[1],
                        ),
                    ).reshape(-1, 1),
                ]
            )
        )

    return D[:m, :n]


def calc_Σ(eigen_vals, A: Optional[sp.Matrix] = None) -> sp.Matrix:
    """Calculates Σ based on the eigen values of A.T @ A."""

    Σ = sp.diag([sp.sqrt(v) for v in eigen_vals], unpack=True)

    if A is not None:
        Σ = pad_or_trim_Σ(Σ, A)

    return Σ


def calc_V(eigen_vects) -> sp.Matrix:
    """Calculates V in SVD of A based on the eigenvectors of A.T @ A."""
    return sp.Matrix(np.hstack([v.normalized() for v in eigen_vects]))


def calc_U(eigen_vects) -> sp.Matrix:
    return calc_V(eigen_vects)


def svd(A):
    eigen_vals, right_eigen_vects = eigen_decompose(A.T @ A)
    Σ = calc_Σ(eigen_vals, A)
    V = calc_V(right_eigen_vects)

    _, left_eigen_vects = eigen_decompose(A @ A.T)
    U = calc_U(left_eigen_vects)
    return U, Σ, V
