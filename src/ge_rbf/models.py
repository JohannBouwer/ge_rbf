"""Radial basis function surrogate models.

One estimator, :class:`RBFRegressor`, covers all three model types from the papers. Which
one you get follows from the data you hand to :meth:`RBFRegressor.fit`:

=========================================  ============================================
Call                                       Model
=========================================  ============================================
``fit(X, y)``                              **FV** — fitted to function values only
``fit(X, y, dy=dy)``                       **GE** — gradient-enhanced, fitted to both
``fit(X, dy=dy, anchor_X=..., anchor_y=...)``  **GO** — gradient-only, anchored
=========================================  ============================================

They share the same machinery: build a kernel matrix, optionally stack the kernel
derivative matrix underneath it, and solve for the weights. The gradient-only model
differs only in that its function-value rows come from a separate set of anchor points
rather than from the sampled locations — a gradient-only fit is otherwise blind to the
constant offset of the function.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import qmc
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError

from ._linalg import check_gradients, check_samples, check_targets, restore_target_shape
from .kernels import Kernel, get_kernel

__all__ = ["RBFRegressor"]

_SOLVERS = ("auto", "solve", "lstsq")


def _draw_centers(
    n_centers: int,
    X: NDArray[np.float64],
    random_state: int | np.random.Generator | None,
) -> NDArray[np.float64]:
    """Draw space-filling centres over the bounding box of ``X``.

    Two choices worth stating, because both are easy to "correct" back into worse ones.

    **The box comes from the samples, not from any declared bounds.** A design of
    experiments does not fill its box in three or more dimensions, so a centre placed in a
    corner nothing was sampled from contributes a column of near-zeros — an
    ill-conditioned direction that buys no expressiveness.

    **The draw happens in whatever coordinates ``fit`` was given**, which is to say after
    any coordinate frame has been applied, because that is where the kernel measures
    distance. A Latin hypercube drawn before a frame and then rotated into it is no longer
    stratified. The cost is that an axis-aligned box around framed samples is larger than
    their rotated hull, so a few centres land in empty corners; least squares copes and
    ``condition_`` reports it.
    """
    if n_centers < 1:
        raise ValueError(f"centers must be at least 1 when given as a count, got {n_centers}.")

    lower, upper = X.min(axis=0), X.max(axis=0)
    unit = qmc.LatinHypercube(d=X.shape[1], seed=random_state).random(n_centers)

    return lower + unit * (upper - lower)


class RBFRegressor(RegressorMixin, BaseEstimator):
    """Radial basis function surrogate, optionally enhanced with sampled gradients.

    Parameters
    ----------
    kernel : str or Kernel, optional
        Basis function. Only ``"gaussian"`` is built in.
    epsilon : float, optional
        Shape parameter of the basis function. This is the hyperparameter the routines in
        :mod:`ge_rbf.selection` search over.
    centers : int or array_like of shape (n_centres, n_features), optional
        Basis function centres, in one of three forms.

        ``None`` (the default) places one centre at every sample, which makes the
        function-value system square and interpolating.

        An **integer** draws that many space-filling centres (Latin hypercube) over the
        bounding box of the samples, at fit time. This decouples where the response is
        *sampled* from where it is *represented*, which matters when the samples are
        clustered — trajectory data strung along a few paths otherwise hands the basis its
        own anisotropy, packing centres tightly along each path and leaving them sparse
        across designs, so neighbouring centres look nearly identical and the system is ill
        conditioned by construction. Use ``random_state`` to make the draw reproducible,
        and :func:`~ge_rbf.selection.basis_search` to choose the count.

        An **array** uses those centres verbatim.

        Anything other than one centre per sample makes the fit a regression rather than an
        interpolation, solved in the least-squares sense, so it no longer has to pass
        exactly through samples that carry solver tolerance in them.
    gradient_weight : float or {"auto"}, optional
        Relative weight of the gradient rows in a gradient-enhanced or gradient-only fit.
        ``None`` (the default) weights them equally with the function rows, which is the
        behaviour used to produce the published results. ``"auto"`` rescales the gradient
        block by ``mean|y| / mean|dy|`` so that neither block dominates the least-squares
        solution purely because of its units.
    solver : {"auto", "solve", "lstsq"}, optional
        ``"auto"`` uses a direct solve when the system is square and least squares
        otherwise. ``"lstsq"`` forces least squares, which is more robust on
        ill-conditioned systems; ``"solve"`` requires a square system and raises otherwise.
    random_state : int or Generator, optional
        Seeds the centre draw. Used only when ``centers`` is an integer, and ignored
        otherwise.

    Attributes
    ----------
    coef_ : ndarray of shape (n_centres,)
        Fitted basis function weights.
    centers_ : ndarray of shape (n_centres, n_features)
        The centres actually used, including those drawn from an integer ``centers``.
    condition_ : float
        Condition number of the (possibly stacked) system matrix. Large values mean the
        fit is numerically delicate; the search routines in :mod:`ge_rbf.selection` use it
        to reject shape parameters.
    mode_ : {"fv", "ge", "go"}
        Which model was fitted, inferred from the arguments to :meth:`fit`.
    n_features_in_ : int

    Examples
    --------
    >>> import numpy as np
    >>> from ge_rbf import RBFRegressor
    >>> X = np.linspace(0, 1, 9).reshape(-1, 1)
    >>> y = np.sin(3 * np.pi * X).ravel()
    >>> dy = 3 * np.pi * np.cos(3 * np.pi * X)
    >>> model = RBFRegressor(epsilon=2.0).fit(X, y, dy=dy)
    >>> y_hat, dy_hat = model.predict(X, return_gradient=True)
    """

    def __init__(
        self,
        kernel: str | Kernel = "gaussian",
        epsilon: float = 1.0,
        centers: ArrayLike | None = None,
        gradient_weight: float | str | None = None,
        solver: str = "auto",
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        self.kernel = kernel
        self.epsilon = epsilon
        self.centers = centers
        self.gradient_weight = gradient_weight
        self.solver = solver
        self.random_state = random_state

    # ------------------------------------------------------------------ fitting

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        dy: ArrayLike | None = None,
        anchor_X: ArrayLike | None = None,
        anchor_y: ArrayLike | None = None,
    ) -> RBFRegressor:
        """Fit the surrogate.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Sample locations.
        y : array_like of shape (n_samples,) or (n_samples, 1), optional
            Sampled function values. Required except in gradient-only mode.
        dy : array_like of shape (n_samples, n_features), optional
            Sampled gradients. Supplying these gives a gradient-enhanced model. Note that
            a gradient-enhanced system has ``n_samples * (1 + n_features)`` rows, so more
            centres than samples is legitimate here and is not rejected.
        anchor_X, anchor_y : array_like, optional
            Locations and function values used to anchor a gradient-only model. Gradients
            alone determine the surrogate only up to an additive constant, so at least one
            function value is needed to pin it down. The anchor need not be a sampled
            point — ``anchor_X=[[0, 0]], anchor_y=[0.0]`` is a legitimate choice.

        Returns
        -------
        self
        """
        if self.solver not in _SOLVERS:
            raise ValueError(f"solver must be one of {_SOLVERS}, got {self.solver!r}.")

        X = check_samples(X)
        n_samples, n_features = X.shape

        mode = self._infer_mode(y, dy, anchor_X, anchor_y)

        if self.centers is None:
            centers = X
        elif np.ndim(self.centers) == 0:
            centers = _draw_centers(int(self.centers), X, self.random_state)
        else:
            centers = check_samples(self.centers, name="centers")

        if centers.shape[1] != n_features:
            raise ValueError(f"centers have {centers.shape[1]} features but X has {n_features}.")

        kernel = get_kernel(self.kernel, self.epsilon)

        # --- function-value block -------------------------------------------------
        if mode == "go":
            anchor_X = check_samples(anchor_X, name="anchor_X")
            if anchor_X.shape[1] != n_features:
                raise ValueError(
                    f"anchor_X has {anchor_X.shape[1]} features but X has {n_features}."
                )
            targets, was_column = check_targets(anchor_y, anchor_X.shape[0], name="anchor_y")
            value_block = kernel(anchor_X, centers)
        else:
            targets, was_column = check_targets(y, n_samples)
            value_block = kernel(X, centers)

        blocks = [value_block]
        right_hand_side = [targets]

        # --- gradient block -------------------------------------------------------
        if mode in ("ge", "go"):
            gradients = check_gradients(dy, n_samples, n_features)
            weight = self._resolve_gradient_weight(targets, gradients)

            blocks.append(weight * kernel.gradient(X, centers))
            # Fortran order: this must match the feature-major row ordering documented on
            # Kernel.gradient, or gradient observations land on the wrong rows.
            right_hand_side.append(weight * gradients.ravel(order="F"))

        system = np.vstack(blocks)
        rhs = np.concatenate(right_hand_side)

        self.coef_, self.condition_ = self._solve(system, rhs)
        self.centers_ = centers
        self.mode_ = mode
        self.n_features_in_ = n_features
        self._targets_were_column = was_column

        return self

    @staticmethod
    def _infer_mode(y, dy, anchor_X, anchor_y) -> str:
        """Work out which of the three models the caller is asking for."""
        anchored = anchor_X is not None or anchor_y is not None

        if anchored:
            if anchor_X is None or anchor_y is None:
                raise ValueError("anchor_X and anchor_y must be supplied together.")
            if dy is None:
                raise ValueError(
                    "A gradient-only model needs gradients: pass dy alongside the anchor."
                )
            if y is not None:
                raise ValueError(
                    "Pass either y (gradient-enhanced) or anchor_X/anchor_y "
                    "(gradient-only), not both."
                )
            return "go"

        if y is None:
            raise ValueError(
                "y is required unless you are fitting a gradient-only model, which needs "
                "anchor_X and anchor_y."
            )

        return "fv" if dy is None else "ge"

    def _resolve_gradient_weight(
        self, targets: NDArray[np.float64], gradients: NDArray[np.float64]
    ) -> float:
        if self.gradient_weight is None:
            return 1.0

        if self.gradient_weight == "auto":
            gradient_scale = np.mean(np.abs(gradients))
            target_scale = np.mean(np.abs(targets))
            if gradient_scale == 0 or target_scale == 0:
                return 1.0
            return float(target_scale / gradient_scale)

        weight = float(self.gradient_weight)
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError(
                f"gradient_weight must be positive, 'auto', or None, got {self.gradient_weight!r}."
            )
        return weight

    def _solve(
        self, system: NDArray[np.float64], rhs: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], float]:
        """Solve for the weights, returning them with the system's condition number.

        The condition number comes from the solve itself wherever possible. ``lstsq``
        already computes the singular values and hands them back, so its 2-norm condition
        number is ``s[0] / s[-1]`` at no extra cost — a separate ``np.linalg.cond`` call
        would repeat the decomposition, which on the stacked gradient-enhanced systems is
        the single most expensive thing a fit does.
        """
        square = system.shape[0] == system.shape[1]
        solver = self.solver

        if solver == "auto":
            solver = "solve" if square else "lstsq"

        if solver == "solve":
            if not square:
                raise ValueError(
                    f"solver='solve' needs a square system, but the system is "
                    f"{system.shape[0]}x{system.shape[1]}. Use solver='lstsq' or 'auto'."
                )
            # np.linalg.solve does an LU factorisation, which says nothing about the
            # singular values, so the condition number needs its own decomposition here.
            return np.linalg.solve(system, rhs), float(np.linalg.cond(system))

        coefficients, _, _, singular_values = np.linalg.lstsq(system, rhs, rcond=None)
        smallest = singular_values[-1]

        # A denormal smallest singular value overflows the ratio to inf, which is the right
        # answer — the system is singular to working precision — so the warning numpy would
        # emit is noise on an expected path. The searches read inf as "reject this".
        with np.errstate(over="ignore"):
            condition = np.inf if smallest == 0 else float(singular_values[0] / smallest)

        return coefficients, condition

    # --------------------------------------------------------------- prediction

    def predict(
        self, X: ArrayLike, return_gradient: bool = False
    ) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Predict function values, and optionally gradients, at new locations.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
        return_gradient : bool, optional
            When True, return ``(y, dy)`` instead of just ``y``.

        Returns
        -------
        y : ndarray
            Shape ``(n_samples,)``, or ``(n_samples, 1)`` if the model was fitted on
            column-vector targets.
        dy : ndarray of shape (n_samples, n_features)
            Only when ``return_gradient`` is True.
        """
        X = self._check_predict_input(X)
        kernel = get_kernel(self.kernel, self.epsilon)

        y = restore_target_shape(kernel(X, self.centers_) @ self.coef_, self._targets_were_column)

        if not return_gradient:
            return y

        return y, self._gradient(kernel, X)

    def predict_gradient(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict gradients at new locations, shape ``(n_samples, n_features)``."""
        X = self._check_predict_input(X)
        return self._gradient(get_kernel(self.kernel, self.epsilon), X)

    def _gradient(self, kernel: Kernel, X: NDArray[np.float64]) -> NDArray[np.float64]:
        stacked = kernel.gradient(X, self.centers_) @ self.coef_
        # Undo the feature-major stacking to get one gradient row per sample.
        return stacked.reshape(X.shape[0], self.n_features_in_, order="F")

    def _check_predict_input(self, X: ArrayLike) -> NDArray[np.float64]:
        if not hasattr(self, "coef_"):
            raise NotFittedError(
                f"This {type(self).__name__} is not fitted yet. Call fit before predict."
            )

        X = check_samples(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but this model was fitted on {self.n_features_in_}."
            )
        return X
