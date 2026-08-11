"""Analytic test problems, each returning both function values and gradients.

Every function here takes samples of shape ``(n_samples, n_features)`` and returns

- ``f`` of shape ``(n_samples,)``
- ``grad`` of shape ``(n_samples, n_features)``

The shapes are uniform across all problems. That matters more than it looks: mixing
``(n,)`` and ``(n, 1)`` function values silently turns an error expression like
``y_true - y_pred`` into an ``(n, n)`` outer difference, which quietly corrupts any RMSE
computed from it.

The module is named ``problems`` rather than ``test_problems`` so that pytest's default
``test_*.py`` collection does not try to import it as a test module.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import expm

from ._linalg import check_samples

__all__ = [
    "ackley",
    "beale",
    "non_isotropic",
    "random_rotation",
    "rastrigin",
    "rosenbrock",
    "sphere",
]


def random_rotation(
    n_features: int, rng: np.random.Generator | int | None = None
) -> NDArray[np.float64]:
    r"""Random rotation matrix built with the exponential map, as in the paper (Eq. 34).

    :math:`R = \exp(\pi (A - A^\top))` where the entries of :math:`A` are drawn uniformly
    from :math:`[-0.5, 0.5]`. The matrix exponential of a skew-symmetric matrix is
    orthogonal, so ``R`` is a rotation by construction.

    Parameters
    ----------
    n_features : int
        Dimension of the space.
    rng : Generator or int, optional
        Seed or generator, for reproducibility.

    Returns
    -------
    ndarray of shape (n_features, n_features)
    """
    generator = np.random.default_rng(rng)
    A = generator.random((n_features, n_features)) - 0.5
    return expm(np.pi * (A - A.T))


def non_isotropic(
    X: ArrayLike, rotation: ArrayLike | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Decomposable test function with a deliberately different length scale per direction.

    .. math:: f(x) = \frac{1}{n} \sum_{i=1}^{n} A_i \sin(F_i x_i)

    The amplitudes and frequencies vary with the coordinate direction, so the function is
    anisotropic while remaining decomposable — which means the ideal reference frame is
    known exactly (it is the coordinate frame itself, scaled per axis). That is what makes
    this problem useful for judging a coordinate transformation scheme.

    Supplying ``rotation`` couples the variables: the function is evaluated in the rotated
    frame, so the ideal reference frame becomes the rotation itself and a transformation
    scheme has to recover both a rotation and a scaling.

    Parameters
    ----------
    X : array_like of shape (n_samples, n_features)
        Sample locations, in the *observed* (possibly coupled) coordinate frame.
    rotation : array_like of shape (n_features, n_features), optional
        Orthogonal matrix. ``None`` (the default) means no rotation, giving the
        decomposable form. Use :func:`random_rotation` to generate one.

    Returns
    -------
    f : ndarray of shape (n_samples,)
    grad : ndarray of shape (n_samples, n_features)
    """
    X = check_samples(X)
    n = X.shape[1]

    if rotation is None:
        Z = X
    else:
        rotation = np.asarray(rotation, dtype=np.float64)
        if rotation.shape != (n, n):
            raise ValueError(
                f"rotation must have shape ({n}, {n}) to match the {n} features in X, "
                f"got {rotation.shape}."
            )
        if not np.allclose(rotation.T @ rotation, np.eye(n), atol=1e-10):
            raise ValueError("rotation must be orthogonal (R.T @ R == I).")
        # Rotate into the decomposable frame; the function is defined there.
        Z = X @ rotation

    dimensions = np.arange(1, n + 1).reshape(1, -1)

    # Frequencies rise from pi/2 to 2*pi across the coordinate directions, and amplitudes
    # dip from 3 to 1 in the middle ones. Together these hold the overall complexity of
    # the problem roughly constant as the dimension grows (paper, Eqs. 32 and 33).
    frequencies = (1.5 * np.pi) / (1 + np.exp(10 * (-dimensions + n / 2))) + np.pi / 2
    amplitudes = -2 * np.exp(-(2 / n) * (dimensions - n / 2) ** 2) + 3

    f = (1 / n) * np.sum(amplitudes * np.sin(frequencies * Z), axis=1)
    grad_z = (1 / n) * amplitudes * frequencies * np.cos(frequencies * Z)

    # Chain rule back to the observed frame: with Z = X @ R, df/dX = (df/dZ) @ R.T.
    grad = grad_z if rotation is None else grad_z @ rotation.T

    return f, grad


def rosenbrock(X: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Rosenbrock function.

    .. math:: f(x) = \sum_{j=1}^{n-1} 100 (x_{j+1} - x_j^2)^2 + (1 - x_j)^2

    Returns
    -------
    f : ndarray of shape (n_samples,)
    grad : ndarray of shape (n_samples, n_features)
    """
    X = check_samples(X)
    if X.shape[1] < 2:
        raise ValueError("The Rosenbrock function needs at least 2 variables.")

    head, tail = X[:, :-1], X[:, 1:]
    residual = tail - head**2

    f = np.sum(100 * residual**2 + (1 - head) ** 2, axis=1)

    grad = np.zeros_like(X)
    grad[:, :-1] += -400 * head * residual - 2 * (1 - head)
    grad[:, 1:] += 200 * residual

    return f, grad


def rastrigin(X: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Rastrigin function, :math:`f(x) = 10n + \sum_i x_i^2 - 10 \cos(2 \pi x_i)`.

    Returns
    -------
    f : ndarray of shape (n_samples,)
    grad : ndarray of shape (n_samples, n_features)
    """
    X = check_samples(X)
    A = 10.0

    f = A * X.shape[1] + np.sum(X**2 - A * np.cos(2 * np.pi * X), axis=1)
    grad = 2 * X + 2 * np.pi * A * np.sin(2 * np.pi * X)

    return f, grad


def ackley(X: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Ackley function.

    .. math::

        f(x) = -a \exp\!\left(-b\sqrt{\tfrac{1}{d}\sum_i x_i^2}\right)
               - \exp\!\left(\tfrac{1}{d}\sum_i \cos(c x_i)\right) + a + e

    Returns
    -------
    f : ndarray of shape (n_samples,)
    grad : ndarray of shape (n_samples, n_features)
        The gradient of the first term is directionally undefined at the origin (the
        square root has a kink there); it is reported as zero at exactly ``x = 0``.
    """
    X = check_samples(X)
    a, b, c = 20.0, 0.2, 2 * np.pi
    d = X.shape[1]

    sum_squares = np.sum(X**2, axis=1)
    sum_cosines = np.sum(np.cos(c * X), axis=1)

    root = np.sqrt(sum_squares / d)
    exp_root = np.exp(-b * root)
    exp_cos = np.exp(sum_cosines / d)

    f = -a * exp_root - exp_cos + a + np.e

    # d/dx_i of -a*exp(-b*u) with u = sqrt(sum_squares/d) is a*b*exp(-b*u)*x_i/(d*u).
    # Note the 1/u: differentiating the square root contributes a factor the earlier
    # implementation of this function omitted.
    with np.errstate(divide="ignore", invalid="ignore"):
        radial = np.where(root > 0, a * b * exp_root / (d * root), 0.0)

    grad = radial[:, None] * X + (c / d) * exp_cos[:, None] * np.sin(c * X)

    return f, grad


def sphere(X: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Sphere function, :math:`f(x) = \sum_i x_i^2`.

    Returns
    -------
    f : ndarray of shape (n_samples,)
    grad : ndarray of shape (n_samples, n_features)
    """
    X = check_samples(X)
    return np.sum(X**2, axis=1), 2 * X


def beale(X: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Beale function, defined for 2 variables only.

    Returns
    -------
    f : ndarray of shape (n_samples,)
    grad : ndarray of shape (n_samples, 2)
    """
    X = check_samples(X)
    if X.shape[1] != 2:
        raise ValueError(f"The Beale function is only defined for 2 variables, got {X.shape[1]}.")

    x1, x2 = X[:, 0], X[:, 1]
    t1 = 1.5 - x1 + x1 * x2
    t2 = 2.25 - x1 + x1 * x2**2
    t3 = 2.625 - x1 + x1 * x2**3

    f = t1**2 + t2**2 + t3**2

    df_dx1 = 2 * t1 * (x2 - 1) + 2 * t2 * (x2**2 - 1) + 2 * t3 * (x2**3 - 1)
    df_dx2 = 2 * t1 * x1 + 2 * t2 * (2 * x1 * x2) + 2 * t3 * (3 * x1 * x2**2)

    return f, np.column_stack([df_dx1, df_dx2])
