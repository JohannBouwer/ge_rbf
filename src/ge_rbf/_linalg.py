"""Array validation and linear-algebra helpers shared across the package.

The functions here exist to make two conventions explicit and enforce them in one
place:

1. **Array shapes.** Samples are always ``(n_samples, n_features)``, function values
   ``(n_samples,)`` and gradients ``(n_samples, n_features)``. Callers may pass column
   vectors for function values; they are flattened on the way in and the original shape
   is restored on the way out (see :func:`check_targets` and :func:`restore_target_shape`).

2. **Eigendecompositions.** Every symmetric matrix in this package is decomposed with
   :func:`symmetric_eigh`, which sorts eigenvalues in descending order and fixes the sign
   of each eigenvector. Without this the decomposition is only defined up to a permutation
   and a sign per column, which makes results irreproducible across NumPy/LAPACK versions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "check_gradients",
    "check_samples",
    "check_targets",
    "restore_target_shape",
    "sqrt_eigenvalues",
    "symmetric_eigh",
]


def check_samples(X: ArrayLike, *, name: str = "X") -> NDArray[np.float64]:
    """Validate a sample matrix and return it as a contiguous ``(n_samples, n_features)`` array.

    Parameters
    ----------
    X : array_like
        Sample locations. A 1-D input is interpreted as a single feature, i.e. it is
        reshaped to ``(n_samples, 1)``.
    name : str, optional
        Name used in error messages.

    Returns
    -------
    ndarray of shape (n_samples, n_features)
    """
    array = np.asarray(X, dtype=np.float64)

    if array.ndim == 1:
        array = array.reshape(-1, 1)

    if array.ndim != 2:
        raise ValueError(f"{name} must be 1- or 2-dimensional, got {array.ndim} dimensions.")

    if array.size == 0:
        raise ValueError(f"{name} must contain at least one sample.")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")

    return np.ascontiguousarray(array)


def check_targets(
    y: ArrayLike, n_samples: int, *, name: str = "y"
) -> tuple[NDArray[np.float64], bool]:
    """Validate function values and flatten them to 1-D.

    Function values are accepted as ``(n_samples,)`` or ``(n_samples, 1)``. The second
    return value records which form was supplied so predictions can be handed back in the
    same shape by :func:`restore_target_shape`.

    Returns
    -------
    values : ndarray of shape (n_samples,)
    was_column : bool
        True when the input was a ``(n_samples, 1)`` column vector.
    """
    array = np.asarray(y, dtype=np.float64)

    was_column = array.ndim == 2 and array.shape[1] == 1
    if was_column:
        array = array.ravel()

    if array.ndim != 1:
        raise ValueError(
            f"{name} must have shape ({n_samples},) or ({n_samples}, 1), got {array.shape}."
        )

    if array.shape[0] != n_samples:
        raise ValueError(f"{name} has {array.shape[0]} entries but there are {n_samples} samples.")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")

    return array, was_column


def restore_target_shape(values: NDArray[np.float64], was_column: bool) -> NDArray[np.float64]:
    """Return ``values`` as a column vector when the training targets were columns."""
    return values.reshape(-1, 1) if was_column else values


def check_gradients(
    dy: ArrayLike, n_samples: int, n_features: int, *, name: str = "dy"
) -> NDArray[np.float64]:
    """Validate a gradient matrix and return it as ``(n_samples, n_features)``."""
    array = np.asarray(dy, dtype=np.float64)

    if array.ndim == 1 and n_features == 1:
        array = array.reshape(-1, 1)

    if array.shape != (n_samples, n_features):
        raise ValueError(f"{name} must have shape ({n_samples}, {n_features}), got {array.shape}.")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")

    return np.ascontiguousarray(array)


def symmetric_eigh(matrix: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Eigendecompose a symmetric matrix with a deterministic, reproducible convention.

    The matrix is symmetrised before decomposition (it should already be symmetric up to
    round-off), eigenvalues are returned in *descending* order, and each eigenvector is
    sign-flipped so that its largest-magnitude component is positive.

    ``eigh`` is used rather than ``eig`` because the latter treats the matrix as general
    and can return eigenvalues with spurious imaginary parts of order machine epsilon.

    Returns
    -------
    eigenvalues : ndarray of shape (n,)
        Descending.
    eigenvectors : ndarray of shape (n, n)
        Columns are the eigenvectors, matching ``eigenvalues`` by position.
    """
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Fix the sign of each column: the entry of largest magnitude is made positive.
    dominant = np.argmax(np.abs(eigenvectors), axis=0)
    signs = np.sign(eigenvectors[dominant, np.arange(eigenvectors.shape[1])])
    signs[signs == 0] = 1.0

    return eigenvalues, eigenvectors * signs


def sqrt_eigenvalues(eigenvalues: NDArray[np.float64]) -> NDArray[np.float64]:
    """Square root of curvature eigenvalues, used as per-direction coordinate scalers.

    Guards against the two ways this can silently produce ``nan``: negative eigenvalues
    (the element-wise median of positive semi-definite matrices is not itself guaranteed
    to be positive semi-definite) and eigenvalues at exactly zero (a direction along which
    the curvature estimate is degenerate, which would collapse that axis entirely).
    """
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)

    if np.any(eigenvalues < 0):
        negative = eigenvalues[eigenvalues < 0]
        raise ValueError(
            "Curvature estimate has negative eigenvalues "
            f"(smallest {negative.min():.3e}); it is not positive semi-definite, so the "
            "coordinate scalers sqrt(lambda) are undefined."
        )

    if np.any(eigenvalues == 0):
        raise ValueError(
            "Curvature estimate has a zero eigenvalue, so at least one coordinate "
            "direction would be scaled to zero. This usually means there are too few "
            "samples to estimate curvature in every direction."
        )

    return np.sqrt(eigenvalues)
