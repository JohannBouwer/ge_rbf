"""Strategies for choosing the shape parameter of a radial basis function surrogate.

The shape parameter controls how wide the basis functions are, and it is the one
hyperparameter that matters. Too wide and the kernel matrix becomes so ill-conditioned
that the fitted weights are numerical noise; too narrow and the surrogate collapses into
spikes around the samples with nothing sensible in between. All three strategies here
sweep a range of candidates and pick the one with the lowest error, subject to a ceiling
on the condition number of the system being solved.

They differ in where the error signal comes from:

:func:`kfold_search`
    Holds out folds of the sampled data in turn. The general-purpose choice.
:func:`validation_search`
    Uses a separate validation set. Best when extra function evaluations are cheap.
:func:`gradient_search`
    Uses the sampled gradients as the validation signal, comparing them against the
    gradients the surrogate predicts at the sample locations. Needs no held-out data at
    all, which is what makes it attractive when samples are expensive.

Each returns a :class:`SearchResult` rather than mutating anything. This is a deliberate
change: the previous implementation returned the best shape parameter but left the model
fitted at the *last* value it happened to try, so callers who trusted the model object
were silently working with a mis-fitted surrogate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import clone
from sklearn.model_selection import KFold

from ._linalg import check_gradients, check_samples, check_targets

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes

    from .models import RBFRegressor

__all__ = [
    "DEFAULT_EPSILONS",
    "SearchResult",
    "gradient_search",
    "kfold_search",
    "plot_search",
    "validation_search",
]

DEFAULT_EPSILONS = np.logspace(-2, 1, 50)


@dataclass(frozen=True)
class SearchResult:
    """Outcome of a shape parameter search.

    Attributes
    ----------
    best_epsilon : float
        The selected shape parameter.
    epsilons : ndarray
        Candidates that passed the conditioning filter, ascending.
    scores : ndarray
        Error for each entry of ``epsilons``. Lower is better; the units depend on which
        search produced it, so compare within a result, not across results.
    conditions : ndarray
        Condition number of the full-data system at each entry of ``epsilons``.
    best_estimator : RBFRegressor or None
        A model fitted on all the data at ``best_epsilon``, unless ``refit=False``.
    rejected : int
        How many candidates were discarded for exceeding ``max_condition``.
    """

    best_epsilon: float
    epsilons: NDArray[np.float64]
    scores: NDArray[np.float64]
    conditions: NDArray[np.float64]
    best_estimator: RBFRegressor | None
    rejected: int

    def __repr__(self) -> str:
        return (
            f"SearchResult(best_epsilon={self.best_epsilon:.4g}, "
            f"best_score={self.scores.min():.4g}, "
            f"n_candidates={self.epsilons.size}, rejected={self.rejected})"
        )


def kfold_search(
    estimator: RBFRegressor,
    X: ArrayLike,
    y: ArrayLike | None = None,
    dy: ArrayLike | None = None,
    *,
    k: int = 5,
    epsilons: ArrayLike = DEFAULT_EPSILONS,
    max_condition: float = 1e13,
    refit: bool = True,
    random_state: int | np.random.RandomState | None = None,
) -> SearchResult:
    """Select a shape parameter by k-fold cross-validation.

    Parameters
    ----------
    estimator : RBFRegressor
        Template. It is cloned for every candidate, never modified.
    X : array_like of shape (n_samples, n_features)
    y : array_like of shape (n_samples,)
    dy : array_like of shape (n_samples, n_features), optional
        When supplied, gradient-enhanced models are fitted and the fold error combines a
        function-value term and a gradient term. The function term is rescaled by
        ``mean|dy| / mean|y|`` so that neither dominates purely because of its units.
    k : int, optional
        Number of folds. Every sample lands in exactly one validation fold, including any
        left over when ``n_samples`` is not divisible by ``k``.
    epsilons : array_like, optional
        Candidate shape parameters.
    max_condition : float, optional
        Candidates whose full-data system is worse conditioned than this are rejected.
    refit : bool, optional
        Fit a model on all the data at the winning shape parameter and return it as
        ``best_estimator``.
    random_state : int or RandomState, optional
        Seeds the fold shuffle, so a search is reproducible.

    Returns
    -------
    SearchResult
    """
    X = check_samples(X)
    n_samples = X.shape[0]
    y_values, _ = check_targets(y, n_samples)
    gradients = None if dy is None else check_gradients(dy, n_samples, X.shape[1])

    if not 2 <= k <= n_samples:
        raise ValueError(f"k must be between 2 and n_samples={n_samples}, got {k}.")

    splitter = KFold(n_splits=k, shuffle=True, random_state=random_state)
    folds = list(splitter.split(X))

    def score(epsilon: float) -> float:
        errors = []
        for train, validate in folds:
            train_gradients = None if gradients is None else gradients[train]
            fitted = _fit(estimator, epsilon, X[train], y_values[train], train_gradients)
            predicted = fitted.predict(X[validate])

            if gradients is None:
                errors.append(_rmse(y_values[validate], predicted))
            else:
                # Put the two error terms on a comparable footing before adding them.
                balance = np.mean(np.abs(gradients[train])) / np.mean(np.abs(y_values[train]))
                errors.append(
                    balance * _rmse(y_values[validate], predicted)
                    + _rmse(gradients[validate], fitted.predict_gradient(X[validate]))
                )
        return float(np.mean(errors))

    return _search(estimator, X, y_values, gradients, epsilons, max_condition, refit, score)


def validation_search(
    estimator: RBFRegressor,
    X: ArrayLike,
    y: ArrayLike,
    X_valid: ArrayLike,
    y_valid: ArrayLike,
    dy: ArrayLike | None = None,
    *,
    epsilons: ArrayLike = DEFAULT_EPSILONS,
    max_condition: float = 1e12,
    refit: bool = True,
) -> SearchResult:
    """Select a shape parameter using the relative error on a separate validation set.

    Parameters
    ----------
    estimator : RBFRegressor
    X, y : array_like
        Training samples and function values.
    X_valid, y_valid : array_like
        Validation locations and the true function values there.
    dy : array_like, optional
        Training gradients; supplying them fits gradient-enhanced models.
    epsilons, max_condition, refit
        As for :func:`kfold_search`.

    Returns
    -------
    SearchResult
    """
    X = check_samples(X)
    y_values, _ = check_targets(y, X.shape[0])
    gradients = None if dy is None else check_gradients(dy, X.shape[0], X.shape[1])

    X_valid = check_samples(X_valid, name="X_valid")
    y_valid_values, _ = check_targets(y_valid, X_valid.shape[0], name="y_valid")

    def score(epsilon: float) -> float:
        fitted = _fit(estimator, epsilon, X, y_values, gradients)
        return _relative_error(y_valid_values, fitted.predict(X_valid))

    return _search(estimator, X, y_values, gradients, epsilons, max_condition, refit, score)


def gradient_search(
    estimator: RBFRegressor,
    X: ArrayLike,
    y: ArrayLike,
    dy: ArrayLike,
    *,
    use_gradients_in_fit: bool = True,
    epsilons: ArrayLike = DEFAULT_EPSILONS,
    max_condition: float = 1e13,
    refit: bool = True,
) -> SearchResult:
    """Select a shape parameter by how well the surrogate reproduces the sampled gradients.

    No data is held out. The error is the relative norm of the difference between the
    sampled gradients and the gradients the fitted surrogate predicts at the sample
    locations.

    Parameters
    ----------
    estimator : RBFRegressor
    X, y, dy : array_like
        Samples, function values and gradients.
    use_gradients_in_fit : bool, optional
        When True (the default) the candidate models are gradient-enhanced, so the
        gradients are used both to fit and to score. When False the models are fitted to
        function values only and the gradients act purely as an independent check — which
        makes this a genuine validation signal rather than a training residual.
    epsilons, max_condition, refit
        As for :func:`kfold_search`.

    Returns
    -------
    SearchResult
    """
    X = check_samples(X)
    y_values, _ = check_targets(y, X.shape[0])
    gradients = check_gradients(dy, X.shape[0], X.shape[1])

    fit_gradients = gradients if use_gradients_in_fit else None

    def score(epsilon: float) -> float:
        fitted = _fit(estimator, epsilon, X, y_values, fit_gradients)
        return _relative_error(gradients, fitted.predict_gradient(X))

    return _search(estimator, X, y_values, fit_gradients, epsilons, max_condition, refit, score)


# --------------------------------------------------------------------------- internals


def _fit(
    estimator: RBFRegressor,
    epsilon: float,
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    gradients: NDArray[np.float64] | None,
) -> RBFRegressor:
    """Clone the template at ``epsilon`` and fit it to the given data.

    Cloning is what keeps a search side-effect free: the caller's estimator is a template
    that is never fitted or reparameterised.

    When ``centers`` is left at its default, a model fitted on a fold is centred on that
    fold's own training points, matching how the folds were originally handled. An
    explicit centre set is shared across folds unchanged.
    """
    candidate = clone(estimator)
    candidate.set_params(epsilon=float(epsilon))
    return candidate.fit(X, y, dy=gradients)


def _search(
    estimator: RBFRegressor,
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    gradients: NDArray[np.float64] | None,
    epsilons: ArrayLike,
    max_condition: float,
    refit: bool,
    score: Any,
) -> SearchResult:
    """Score every candidate, drop the badly conditioned ones, and pick the best."""
    epsilons = np.sort(np.asarray(epsilons, dtype=np.float64).ravel())
    if epsilons.size == 0:
        raise ValueError("epsilons must contain at least one candidate.")
    if np.any(epsilons <= 0):
        raise ValueError("epsilons must all be positive.")
    if max_condition <= 0:
        raise ValueError(f"max_condition must be positive, got {max_condition}.")

    scores = np.empty(epsilons.size)
    conditions = np.empty(epsilons.size)

    for i, epsilon in enumerate(epsilons):
        scores[i] = score(epsilon)
        conditions[i] = _fit(estimator, epsilon, X, y, gradients).condition_

    # Reject ill-conditioned candidates *before* choosing the winner. The previous
    # implementation applied this filter only when drawing the diagnostic plot, so
    # max_condition had no effect on the value it returned.
    keep = conditions <= max_condition
    rejected = int((~keep).sum())

    if not keep.any():
        raise ValueError(
            f"Every candidate shape parameter produced a system worse conditioned than "
            f"max_condition={max_condition:.3g} (best was {conditions.min():.3g}). Raise "
            f"max_condition, or try larger shape parameters — wide basis functions are "
            f"what make the system ill-conditioned."
        )

    epsilons, scores, conditions = epsilons[keep], scores[keep], conditions[keep]
    best_epsilon = float(epsilons[np.argmin(scores)])

    best_estimator = None
    if refit:
        best_estimator = _fit(estimator, best_epsilon, X, y, gradients)

    return SearchResult(
        best_epsilon=best_epsilon,
        epsilons=epsilons,
        scores=scores,
        conditions=conditions,
        best_estimator=best_estimator,
        rejected=rejected,
    )


def _rmse(actual: NDArray[np.float64], predicted: NDArray[np.float64]) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def _relative_error(actual: NDArray[np.float64], predicted: NDArray[np.float64]) -> float:
    denominator = np.linalg.norm(actual)
    if denominator == 0:
        return float(np.linalg.norm(actual - predicted))
    return float(np.linalg.norm(actual - predicted) / denominator)


def plot_search(result: SearchResult, ax: Axes | None = None) -> Axes:
    """Plot search error against shape parameter on log axes, marking the winner.

    Matplotlib is imported here rather than at module scope so that importing
    :mod:`ge_rbf` does not pull in a plotting stack.

    Parameters
    ----------
    result : SearchResult
    ax : matplotlib Axes, optional
        Draw on these axes instead of creating a new figure.

    Returns
    -------
    matplotlib Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    ax.loglog(result.epsilons, result.scores, "-", label="Search error")
    ax.loglog(result.best_epsilon, result.scores.min(), "k.", markersize=12, label="Selected")
    ax.set_xlabel("Shape parameter")
    ax.set_ylabel("Error")
    ax.set_title("Error vs shape parameter")
    ax.legend()

    return ax
