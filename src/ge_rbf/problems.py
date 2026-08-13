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

:func:`load_path_samples` and :func:`load_path` are the exception to the uniform signature
above: together they stand in for a *family of solved paths* rather than a field sampled at
scattered points, which is a different shape of data and needs its own sampling geometry.
See :mod:`ge_rbf.trajectories`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import expm
from scipy.stats import qmc

from ._linalg import check_samples

__all__ = [
    "ackley",
    "beale",
    "load_path",
    "load_path_samples",
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


def load_path_samples(
    n_designs: int = 15,
    n_steps: int = 18,
    n_variables: int = 3,
    *,
    arc_length: float = 1.0,
    ragged: float = 0.25,
    rng: np.random.Generator | int | None = 0,
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Sampling geometry of a family of solved paths.

    A continuation or transient solver does not return scattered points. It returns a
    handful of *trajectories*: one per design, each a chain of closely spaced states
    parametrised by a time-like coordinate. Stacked into one array that is a very
    particular clustering, and it is the clustering — not the response — that breaks the
    usual defaults. This function produces the geometry with no response attached, so the
    two can be varied independently.

    Designs are drawn by Latin hypercube on the unit cube, so any anisotropy in the layout
    is the doing of the trajectory axis alone.

    Parameters
    ----------
    n_designs : int, optional
        Number of trajectories.
    n_steps : int, optional
        Stored points per trajectory.
    n_variables : int, optional
        Number of design variables. The returned array has one more column than this.
    arc_length : float, optional
        Nominal length of a trajectory, in whatever units the trajectory coordinate is
        measured in. Only the *ratio* of this to the design spacing matters to anything
        downstream, so changing it is a units change and nothing more.
    ragged : float, optional
        Fractional spread of trajectory lengths about ``arc_length``, so paths end at
        different points as real ones do. ``0`` makes every path the same length.
    rng : Generator or int, optional
        Seed or generator, for reproducibility.

    Returns
    -------
    Z : ndarray of shape (n_designs * n_steps, n_variables + 1)
        One row per stored point. The design variables come first and the **trajectory
        coordinate is the last column**, which is what
        :class:`~ge_rbf.trajectories.TrajectoryScaler` assumes by default.
    groups : ndarray of shape (n_designs * n_steps,)
        Which trajectory each row belongs to.

    Notes
    -----
    At the defaults this reproduces the layout measured on real load-path data: the mean
    nearest-neighbour distance between designs is several times the gap between
    consecutive points along one path, so every point's nearest neighbours are the points
    before and after it on its own trajectory and nothing else.

    Examples
    --------
    >>> from ge_rbf.problems import load_path, load_path_samples
    >>> Z, groups = load_path_samples()
    >>> y, dy = load_path(Z)
    >>> Z.shape, y.shape, dy.shape
    ((270, 4), (270,), (270, 4))
    """
    if n_designs < 1 or n_steps < 1 or n_variables < 1:
        raise ValueError(
            f"n_designs, n_steps and n_variables must all be at least 1, got "
            f"{n_designs}, {n_steps} and {n_variables}."
        )
    if arc_length <= 0:
        raise ValueError(f"arc_length must be positive, got {arc_length}.")
    if not 0 <= ragged < 1:
        raise ValueError(f"ragged must be in [0, 1), got {ragged}.")

    generator = np.random.default_rng(rng)
    designs = qmc.LatinHypercube(d=n_variables, seed=generator).random(n_designs)

    # Each path runs from one step in to its own end point, so no two paths share their
    # trajectory coordinates exactly.
    lengths = arc_length * (1 + ragged * (2 * generator.random(n_designs) - 1))
    steps = np.concatenate([np.linspace(L / n_steps, L, n_steps) for L in lengths])

    Z = np.column_stack([np.repeat(designs, n_steps, axis=0), steps])
    groups = np.repeat(np.arange(n_designs), n_steps)

    return Z, groups


def load_path(
    Z: ArrayLike,
    *,
    arc_length: float = 1.0,
    n_periods: float = 0.75,
    rotation: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""A response traced along a load path, over the design variables and the path together.

    .. math:: f(x, s) = A(x) \sin(\omega(x) s) + g(x)

    with :math:`A(x) = 1 + \kappa\,\overline{x}`, :math:`\omega(x) = \omega_0 (1 + \beta x_0)`
    and :math:`g` the package's own anisotropic test function :func:`non_isotropic`.
    Constants are :math:`\kappa = 0.5`, :math:`\beta = 0.3` and
    :math:`\omega_0 = 2\pi\,\texttt{n\_periods} / \texttt{arc\_length}`.

    Three features make this harder than a smooth field, and each is there for a reason:

    **A limit point at a design-dependent location.** :math:`\partial f/\partial s = 0` at
    :math:`\omega s = \pi/2`, and because :math:`\omega` varies with :math:`x_0` that peak
    sits at a different point along every path. Past :math:`\omega s = \pi` the curvature
    in :math:`s` changes sign, so the local Hessians are genuinely indefinite over part of
    the domain — which is exactly when ``"ge-lhm"`` cannot produce a frame and has to
    retreat.

    **Curvature coupling between a design axis and the path axis.** Because
    :math:`\omega` depends on :math:`x_0`, :math:`\partial^2 f / \partial x_0 \partial s`
    is non-zero: there is a genuine rotation mixing the two for a transformation scheme to
    find. Without it, scaling the trajectory axis would be cosmetic.

    **Anisotropy across the design variables**, inherited from :func:`non_isotropic`, whose
    ideal frame is known. Passing a ``rotation`` couples the design variables as well.

    Parameters
    ----------
    Z : array_like of shape (n_samples, n_variables + 1)
        Design variables followed by the trajectory coordinate in the last column, as
        returned by :func:`load_path_samples`.
    arc_length : float, optional
        The nominal path length ``Z`` was generated with. It sets :math:`\omega_0`, so
        passing the same value used for :func:`load_path_samples` keeps the response the
        same shape whatever units the trajectory coordinate is in.
    n_periods : float, optional
        How much of a period of the path oscillation fits in ``arc_length``. Larger values
        put more sign changes of the path curvature inside the sampled range.
    rotation : array_like of shape (n_variables, n_variables), optional
        Passed to :func:`non_isotropic` to couple the design variables. Note it applies to
        the design block only, never to the trajectory coordinate.

    Returns
    -------
    f : ndarray of shape (n_samples,)
    grad : ndarray of shape (n_samples, n_variables + 1)
        Derivatives with respect to the design variables, then with respect to the
        trajectory coordinate — the same column order as ``Z``.
    """
    Z = check_samples(Z, name="Z")
    if Z.shape[1] < 2:
        raise ValueError(
            f"Z needs at least one design variable and a trajectory coordinate, so at "
            f"least 2 columns; got {Z.shape[1]}."
        )
    if arc_length <= 0:
        raise ValueError(f"arc_length must be positive, got {arc_length}.")

    X, s = Z[:, :-1], Z[:, -1]
    n_variables = X.shape[1]

    kappa, beta = 0.5, 0.3
    base_frequency = 2 * np.pi * n_periods / arc_length

    amplitude = 1 + kappa * np.mean(X, axis=1)
    frequency = base_frequency * (1 + beta * X[:, 0])

    g, dg = non_isotropic(X, rotation)

    phase = frequency * s
    f = amplitude * np.sin(phase) + g

    # d(amplitude)/dx_i is kappa/n_variables for every i; d(frequency)/dx_i is non-zero
    # for the first variable only, and it is what couples the design and path axes.
    d_frequency = np.zeros(n_variables)
    d_frequency[0] = base_frequency * beta

    grad_x = (
        (kappa / n_variables) * np.sin(phase)[:, None]
        + (amplitude * s * np.cos(phase))[:, None] * d_frequency
        + dg
    )
    grad_s = amplitude * frequency * np.cos(phase)

    return f, np.column_stack([grad_x, grad_s])


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
