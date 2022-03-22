from optparse import Option
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
        if lambda_ > 0:
            for i in range(multiplicity):
                vals.append(lambda_)
                vects.append(xs[i])

    return vals, vects


def calc_Σ(eigen_vals, A: Optional[sp.Matrix] = None) -> sp.Matrix:
    """Calculates Σ based on the eigen values of A.T @ A."""

    return sp.diag([sp.sqrt(v) for v in eigen_vals], unpack=True)


def calc_V(eigen_vects) -> sp.Matrix:
    """Calculates V in SVD of A based on the eigenvectors of A.T @ A."""
    return sp.Matrix(np.hstack([v.normalized() for v in eigen_vects]))


def calc_U(eigen_vects) -> sp.Matrix:
    return calc_V(eigen_vects)


def svd(A):
    eigen_vals, right_eigen_vects = eigen_decompose(A.T @ A)
    Σ = calc_Σ(eigen_vals, A)
    V = calc_V(right_eigen_vects)

    U = A @ V @ Σ.inv()

    return U, Σ, V
