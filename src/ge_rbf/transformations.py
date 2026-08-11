r"""Coordinate system transformations that make sampled data closer to isotropic.

The basis functions used by radial basis function surrogates are isotropic: they depend
only on :math:`\|x - c\|`, so the model assumes the response varies at the same rate in
every direction. Real responses rarely do. The mismatch shows up as model bias, and it is
why adding gradient information to a surrogate can *fail* to improve it — the extra
information is fighting the functional form rather than informing it.

The fix is a pre-processing step: find a rotation and a per-direction scaling that make
the data manifold roughly isotropic, express the samples and their gradients in that
frame, and fit there. Component-wise scaling alone is not enough when the variables are
coupled; a rotation is needed too.

The transformation is estimated from curvature. Local curvature estimates are collected
around each sample, made positive semi-definite, and combined into one global estimate,
whose eigenvectors give the rotation and whose eigenvalue square roots give the scaling.

Available methods
-----------------
``"ge-lhm"``
    Local Hessians from sampled gradients via symmetric rank-one (SR1) updates. Needs
    ``n + 1`` points per estimate, so it scales linearly with dimension. The recommended
    method when gradients are available.
``"ge-dlhm"``
    As above but keeping only the diagonal, i.e. component-wise scaling with no rotation.
    Useful for showing what is lost by not rotating.
``"fv-lhm"``
    Local Hessians from quadratic fits to function values. Needs
    ``n(n+1)/2 + n + 1`` points per estimate, so it becomes expensive quickly.
``"asm"``
    Active subspace method: the eigendecomposition of the averaged outer product of the
    gradients. A global measure rather than a collection of local ones.
``"ideal"``
    A rotation and scaling you supply, for when the optimal frame is known.

Unlike the original implementation, a transformer never modifies the estimator or data it
is given. It computes a frame and holds it; you apply it explicitly.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

from ._linalg import check_gradients, check_samples, check_targets, sqrt_eigenvalues, symmetric_eigh

__all__ = ["IsotropicTransformer"]

METHODS = ("ge-lhm", "ge-dlhm", "fv-lhm", "asm", "ideal")

# Tolerance below which an SR1 update is considered degenerate and the sweep stops.
_SR1_TOLERANCE = 1e-6

# Local Hessians are rounded before the element-wise median so that near-identical values
# do not reorder unpredictably between platforms.
_MEDIAN_ROUNDING = 10


class IsotropicTransformer(TransformerMixin, BaseEstimator):
    """Estimate and apply a rotation and scaling that make sampled data near-isotropic.

    Parameters
    ----------
    method : {"ge-lhm", "ge-dlhm", "fv-lhm", "asm", "ideal"}, optional
        How the frame is estimated. See the module docstring.
    n_neighbors : int, optional
        Points used per local curvature estimate, for the LHM methods. ``None`` uses the
        minimum each method needs: ``n_features + 1`` for ``"ge-lhm"``, and
        ``n(n+1)/2 + n + 1`` for ``"fv-lhm"``.
    rotation : array_like of shape (n_features, n_features), optional
        Orthogonal matrix, required by ``method="ideal"`` and ignored otherwise.
    scaling : array_like of shape (n_features,), optional
        Per-direction scalers, required by ``method="ideal"`` and ignored otherwise.
        These are used **directly** as the scalers. (The original implementation squared
        rooted whatever was passed here, which made the argument mean "eigenvalues" while
        being documented as "scalers".)

    Attributes
    ----------
    rotation_ : ndarray of shape (n_features, n_features)
        Columns are the directions of the new frame.
    scaling_ : ndarray of shape (n_features,)
        Per-direction scalers, ``sqrt(eigenvalues_)`` for the estimated methods.
    eigenvalues_ : ndarray of shape (n_features,)
        Curvature magnitudes, in descending order.
    curvature_ : ndarray of shape (n_features, n_features)
        The global curvature estimate the frame was derived from: an averaged Hessian for
        the LHM methods, the gradient covariance for ``"asm"``.
    n_features_in_ : int

    Notes
    -----
    Where the implementation departs from the published description, see
    ``_local_hessians_from_gradients`` and the README section "Implementation notes vs.
    the paper".

    Examples
    --------
    >>> import numpy as np
    >>> from ge_rbf import IsotropicTransformer, RBFRegressor
    >>> from ge_rbf.problems import non_isotropic
    >>> X = np.random.default_rng(0).random((30, 2))
    >>> y, dy = non_isotropic(X)
    >>> frame = IsotropicTransformer().fit(X, dy=dy)
    >>> model = RBFRegressor(epsilon=1.0).fit(
    ...     frame.transform(X), y, dy=frame.transform_gradient(dy)
    ... )
    """

    def __init__(
        self,
        method: str = "ge-lhm",
        n_neighbors: int | None = None,
        rotation: ArrayLike | None = None,
        scaling: ArrayLike | None = None,
    ) -> None:
        self.method = method
        self.n_neighbors = n_neighbors
        self.rotation = rotation
        self.scaling = scaling

    # ------------------------------------------------------------------ fitting

    def fit(
        self, X: ArrayLike, y: ArrayLike | None = None, dy: ArrayLike | None = None
    ) -> IsotropicTransformer:
        """Estimate the frame from sampled data.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
        y : array_like of shape (n_samples,), optional
            Function values. Required by ``method="fv-lhm"``.
        dy : array_like of shape (n_samples, n_features), optional
            Gradients. Required by the gradient-based methods.

        Returns
        -------
        self
        """
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}.")

        X = check_samples(X)
        n_features = X.shape[1]
        self.n_features_in_ = n_features

        if self.method == "ideal":
            self._fit_ideal(n_features)
            return self

        if self.method == "asm":
            self.curvature_ = self._gradient_covariance(X, dy)
            eigenvalues, rotation = symmetric_eigh(self.curvature_)

        elif self.method in ("ge-lhm", "ge-dlhm"):
            local = self._local_hessians_from_gradients(X, dy)
            self.curvature_ = self._combine_local_hessians(local)

            if self.method == "ge-dlhm":
                # Component-wise scaling only: keep the diagonal, drop the rotation.
                eigenvalues = np.diag(self.curvature_).copy()
                rotation = np.identity(n_features)
            else:
                eigenvalues, rotation = symmetric_eigh(self.curvature_)

        else:  # fv-lhm
            local = self._local_hessians_from_values(X, y)
            self.curvature_ = self._combine_local_hessians(local)
            eigenvalues, rotation = symmetric_eigh(self.curvature_)

        self.eigenvalues_ = eigenvalues
        self.rotation_ = rotation
        self.scaling_ = sqrt_eigenvalues(eigenvalues)

        return self

    def _fit_ideal(self, n_features: int) -> None:
        if self.rotation is None or self.scaling is None:
            raise ValueError("method='ideal' requires both rotation and scaling.")

        rotation = np.asarray(self.rotation, dtype=np.float64)
        scaling = np.asarray(self.scaling, dtype=np.float64).ravel()

        if rotation.shape != (n_features, n_features):
            raise ValueError(
                f"rotation must have shape ({n_features}, {n_features}), got {rotation.shape}."
            )
        if not np.allclose(rotation.T @ rotation, np.identity(n_features), atol=1e-10):
            raise ValueError("rotation must be orthogonal (R.T @ R == I).")
        if scaling.shape != (n_features,):
            raise ValueError(f"scaling must have shape ({n_features},), got {scaling.shape}.")
        if np.any(scaling <= 0):
            raise ValueError("scaling entries must all be positive.")

        self.rotation_ = rotation
        self.scaling_ = scaling
        self.eigenvalues_ = scaling**2
        self.curvature_ = rotation @ np.diag(self.eigenvalues_) @ rotation.T

    # ------------------------------------------------------- curvature estimation

    @staticmethod
    def _gradient_covariance(X: NDArray[np.float64], dy: ArrayLike | None) -> NDArray[np.float64]:
        """Active subspace method: the averaged outer product of the sampled gradients."""
        if dy is None:
            raise ValueError("method='asm' requires gradients; pass dy.")
        gradients = check_gradients(dy, X.shape[0], X.shape[1])
        return gradients.T @ gradients / gradients.shape[0]

    def _local_hessians_from_gradients(
        self, X: NDArray[np.float64], dy: ArrayLike | None
    ) -> NDArray[np.float64]:
        """One local Hessian per sample, from symmetric rank-one updates over neighbours.

        Each estimate sweeps the closest neighbours from furthest to closest, applying the
        SR1 update (paper, Eq. 21)

        .. math::
            H_{k+1} = H_k + \\frac{(y_k - H_k \\Delta x_k)(y_k - H_k \\Delta x_k)^\\top}
                                   {(y_k - H_k \\Delta x_k)^\\top \\Delta x_k}

        with :math:`y_k` the difference in sampled gradients (Eq. 22).

        Two deliberate departures from the published description, kept so that results
        reproduce the original implementation:

        1. The sweep starts from the outer product of the gradient at the point,
           :math:`\\nabla f \\nabla f^\\top`, rather than from the identity matrix that
           Procedure 2 specifies.
        2. The sweep stops early if any component of the SR1 numerator falls below
           ``1e-6``, guarding against a vanishing denominator. This guard is not in the
           paper and, in testing, never triggered on the problems studied there.
        """
        if dy is None:
            raise ValueError(f"method={self.method!r} requires gradients; pass dy.")

        n_samples, n_features = X.shape
        gradients = check_gradients(dy, n_samples, n_features)

        n_neighbors = n_features + 1 if self.n_neighbors is None else int(self.n_neighbors)
        self._check_neighbour_count(n_neighbors, n_samples)

        # Closest neighbours of each point, excluding the point itself.
        distances = cdist(X, X, metric="euclidean")
        neighbors = np.argsort(distances, axis=1)[:, 1 : n_neighbors + 1]

        hessians = np.empty((n_samples, n_features, n_features))

        for i in range(n_samples):
            hessian = np.outer(gradients[i], gradients[i])

            for j in neighbors[i][::-1]:  # furthest to closest, as in Procedure 2
                delta_x = (X[j] - X[i]).reshape(-1, 1)
                delta_gradient = (gradients[j] - gradients[i]).reshape(-1, 1)

                residual = delta_gradient - hessian @ delta_x

                if np.any(np.abs(residual) < _SR1_TOLERANCE):
                    break

                hessian = hessian + (residual @ residual.T) / (residual.T @ delta_x)

            hessians[i] = hessian

        return hessians

    def _local_hessians_from_values(
        self, X: NDArray[np.float64], y: ArrayLike | None
    ) -> NDArray[np.float64]:
        """One local Hessian per sample, from a quadratic fitted to nearby function values.

        Around each point a full quadratic

        .. math:: f(x + d) = \\sum_i a_i d_i^2 + \\sum_{i<j} b_{ij} d_i d_j
                             + \\sum_i c_i d_i + \\text{const}

        is fitted to the closest ``n(n+1)/2 + n + 1`` samples, and the Hessian read off as
        :math:`H_{ii} = 2 a_i`, :math:`H_{ij} = b_{ij}`.
        """
        if y is None:
            raise ValueError("method='fv-lhm' requires function values; pass y.")

        n_samples, n_features = X.shape
        values, _ = check_targets(y, n_samples)

        upper_i, upper_j = np.triu_indices(n_features, k=1)
        n_cross = upper_i.size
        # squares + cross terms + linear terms + constant
        n_terms = n_features + n_cross + n_features + 1

        n_neighbors = n_terms if self.n_neighbors is None else int(self.n_neighbors)
        if n_neighbors < n_terms:
            raise ValueError(
                f"A quadratic fit in {n_features} dimensions needs at least {n_terms} "
                f"points, but n_neighbors={n_neighbors}."
            )
        if n_neighbors > n_samples:
            raise ValueError(
                f"n_neighbors={n_neighbors} exceeds the {n_samples} available samples. "
                f"A function-value Hessian estimate in {n_features} dimensions needs at "
                f"least {n_terms} samples; consider method='ge-lhm', which needs only "
                f"{n_features + 2}."
            )

        distances = cdist(X, X, metric="euclidean")
        neighbors = np.argsort(distances, axis=1)[:, :n_neighbors]

        hessians = np.empty((n_samples, n_features, n_features))

        for i in range(n_samples):
            offsets = X[neighbors[i]] - X[i]

            design = np.empty((n_neighbors, n_terms))
            design[:, :n_features] = offsets**2
            design[:, n_features : n_features + n_cross] = offsets[:, upper_i] * offsets[:, upper_j]
            design[:, n_features + n_cross : 2 * n_features + n_cross] = offsets
            design[:, -1] = 1.0

            coefficients, *_ = np.linalg.lstsq(design, values[neighbors[i]], rcond=None)

            hessian = np.zeros((n_features, n_features))
            hessian[np.diag_indices(n_features)] = 2 * coefficients[:n_features]
            hessian[upper_i, upper_j] = coefficients[n_features : n_features + n_cross]
            hessian[upper_j, upper_i] = coefficients[n_features : n_features + n_cross]

            hessians[i] = hessian

        return hessians

    @staticmethod
    def _combine_local_hessians(hessians: NDArray[np.float64]) -> NDArray[np.float64]:
        """Reduce a stack of local Hessians to one global curvature estimate.

        Each local Hessian is first rebuilt from the *absolute values* of its eigenvalues
        (paper, Eqs. 18-19). Local curvature can be concave or convex, and averaging the
        two directly can cancel to nothing; taking magnitudes keeps the direction and
        strength of the curvature while discarding its sign.

        The reduction across points is an element-wise **median**, where the paper (Eq. 20)
        specifies the mean. The median is far less sensitive to a single badly conditioned
        local estimate. This is a deliberate divergence — see the README.
        """
        reconstructed = np.empty_like(hessians)

        for i, hessian in enumerate(hessians):
            eigenvalues, eigenvectors = symmetric_eigh(hessian)
            reconstructed[i] = eigenvectors @ np.diag(np.abs(eigenvalues)) @ eigenvectors.T

        return np.median(np.round(reconstructed, _MEDIAN_ROUNDING), axis=0)

    @staticmethod
    def _check_neighbour_count(n_neighbors: int, n_samples: int) -> None:
        if n_neighbors < 1:
            raise ValueError(f"n_neighbors must be at least 1, got {n_neighbors}.")
        if n_neighbors >= n_samples:
            raise ValueError(
                f"n_neighbors={n_neighbors} needs more than {n_samples} samples "
                "(each point's neighbours exclude itself)."
            )

    # ------------------------------------------------------------ applying the frame

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Express samples in the isotropic frame: ``X @ rotation_ * scaling_``."""
        X = self._check_applied_to(X, "X")
        return X @ self.rotation_ * self.scaling_

    def inverse_transform(self, Xt: ArrayLike) -> NDArray[np.float64]:
        """Map samples in the isotropic frame back to the original coordinates."""
        Xt = self._check_applied_to(Xt, "Xt")
        return (Xt / self.scaling_) @ self.rotation_.T

    def transform_gradient(self, dy: ArrayLike) -> NDArray[np.float64]:
        r"""Express gradients in the isotropic frame.

        Rotating and scaling the coordinates changes what a gradient means, so sampled
        gradients must be converted before they are used to fit a model in the new frame.
        With :math:`\hat{x} = S R^\top x`, the chain rule gives
        :math:`\nabla_{\hat{x}} f = S^{-1} R^\top \nabla_x f` (paper, Eq. 29), which in
        the row-vector layout used here is ``dy @ rotation_ / scaling_``.
        """
        dy = self._check_applied_to(dy, "dy")
        return dy @ self.rotation_ / self.scaling_

    def inverse_transform_gradient(self, dyt: ArrayLike) -> NDArray[np.float64]:
        """Map gradients in the isotropic frame back to the original coordinates."""
        dyt = self._check_applied_to(dyt, "dyt")
        return (dyt * self.scaling_) @ self.rotation_.T

    def _check_applied_to(self, array: ArrayLike, name: str) -> NDArray[np.float64]:
        if not hasattr(self, "rotation_"):
            raise NotFittedError(f"This {type(self).__name__} is not fitted yet. Call fit first.")

        array = check_samples(array, name=name)
        if array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"{name} has {array.shape[1]} features, but this transformer was fitted "
                f"on {self.n_features_in_}."
            )
        return array
