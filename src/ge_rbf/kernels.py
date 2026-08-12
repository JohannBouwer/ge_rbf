"""Radial basis functions and their analytic derivatives.

A kernel supplies two matrices: the basis function values themselves, and the derivatives
of those basis functions with respect to each input coordinate. Gradient-enhanced and
gradient-only models stack the two into a single system, so the row ordering of the
derivative matrix is a hard contract — see :meth:`Kernel.gradient`.

Only the Gaussian is implemented. It is the basis function used throughout the papers this
package accompanies, and the one every result was generated with. :class:`Kernel` is the
extension point if the multiquadric or inverse-quadratic alternatives are ever needed;
note that the meaning of ``epsilon`` differs between basis functions, so a new kernel must
document its own convention rather than assume this one.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

__all__ = ["GaussianKernel", "Kernel", "get_kernel"]


@runtime_checkable
class Kernel(Protocol):
    """Interface a basis function must provide."""

    epsilon: float

    def __call__(self, X: NDArray[np.float64], C: NDArray[np.float64]) -> NDArray[np.float64]:
        """Basis function values, shape ``(n_samples, n_centres)``."""
        ...

    def gradient(self, X: NDArray[np.float64], C: NDArray[np.float64]) -> NDArray[np.float64]:
        """Basis function derivatives, shape ``(n_samples * n_features, n_centres)``."""
        ...


class GaussianKernel:
    r"""Gaussian basis function :math:`\phi(x, c) = \exp(-\epsilon \|x - c\|^2)`.

    Note the convention: ``epsilon`` multiplies the *squared* distance, so it has units of
    inverse length squared and larger values give narrower basis functions. This matches
    the form used in the accompanying papers.

    Parameters
    ----------
    epsilon : float
        Shape parameter. Must be positive.
    """

    def __init__(self, epsilon: float = 1.0) -> None:
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError(f"epsilon must be a positive finite number, got {epsilon!r}.")
        self.epsilon = float(epsilon)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(epsilon={self.epsilon!r})"

    def __call__(self, X: NDArray[np.float64], C: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the basis functions.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        C : ndarray of shape (n_centres, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_centres)
            Entry ``(i, j)`` is :math:`\\phi(x_i, c_j)`.
        """
        # Deliberately `euclidean ** 2` rather than the cheaper `sqeuclidean`: the two use
        # different summation orders and disagree in the last few bits, which is enough to
        # perturb coefficients solved from an ill-conditioned kernel matrix. Keeping this
        # form makes results bit-identical to the originally published implementation.
        squared_distance = cdist(X, C, metric="euclidean") ** 2
        return np.exp(-self.epsilon * squared_distance)

    def gradient(self, X: NDArray[np.float64], C: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Derivatives of the basis functions with respect to the input coordinates.

        :math:`\partial \phi(x, c) / \partial x_f = -2 \epsilon (x_f - c_f) \phi(x, c)`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        C : ndarray of shape (n_centres, n_features)

        Returns
        -------
        ndarray of shape (n_samples * n_features, n_centres)
            **Feature-major ordering**: row ``f * n_samples + i`` holds
            :math:`\partial \phi(x_i, c_j) / \partial x_f` across ``j``.

            This is the ordering produced by flattening a gradient array of shape
            ``(n_samples, n_features)`` in Fortran order, which is how gradient
            observations are stacked into the right-hand side of the fitting system. The
            two must agree or the model silently fits gradients to the wrong rows;
            ``tests/test_kernels.py`` pins the convention.
        """
        n_samples, n_features = X.shape
        n_centres = C.shape[0]

        values = self(X, C)

        # (n_features, n_samples, n_centres); broadcasting rather than np.repeat so only
        # one array of this size is ever allocated.
        offsets = np.transpose(X[:, None, :] - C[None, :, :], (2, 0, 1))
        derivatives = (-2.0 * self.epsilon) * offsets * values[None, :, :]

        # C-order flattening of a (n_features, n_samples, ...) array gives feature-major rows.
        return derivatives.reshape(n_features * n_samples, n_centres)


_KERNELS: dict[str, type] = {"gaussian": GaussianKernel}


def get_kernel(kernel: str | Kernel, epsilon: float) -> Kernel:
    """Resolve a kernel name or instance to a kernel configured with ``epsilon``.

    Passing an instance re-creates it at the requested ``epsilon`` so that the shape
    parameter always comes from the estimator, never from a stale kernel object.
    """
    if isinstance(kernel, str):
        try:
            factory = _KERNELS[kernel]
        except KeyError:
            available = ", ".join(sorted(_KERNELS))
            raise ValueError(
                f"Unknown kernel {kernel!r}. Available kernels: {available}."
            ) from None
        return factory(epsilon=epsilon)

    if isinstance(kernel, Kernel):
        return type(kernel)(epsilon=epsilon)

    raise TypeError(f"kernel must be a string or a Kernel instance, got {type(kernel).__name__}.")
