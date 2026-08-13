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

Two things beyond the shape parameter
-------------------------------------
:func:`basis_search` searches the **number of basis functions** alongside the shape
parameter. Once the centres are re-sampled rather than placed one per sample, the count
matters as much as the width, and the two interact — the best shape parameter moves by an
order of magnitude across centre counts, so searching either alone finds the best value
given a bad choice of the other. It is the one search here that holds data out by default,
because the centre count is the one hyperparameter a training-residual score cannot choose;
its docstring gives the measurements.

:func:`scaled_epsilons` makes the candidate shape parameters relative to how far apart the
samples actually are. Absolute candidates are only meaningful when the coordinates happen
to be of order one, and a coordinate frame stretches each axis by its curvature — orders of
magnitude for a stiff response. The failure is quiet: every basis function underflows and
the model predicts a flat zero.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import pdist
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, KFold

from ._linalg import check_gradients, check_samples, check_targets

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes

    from .models import RBFRegressor

__all__ = [
    "DEFAULT_EPSILONS",
    "BasisSearchResult",
    "SearchResult",
    "basis_search",
    "gradient_search",
    "kfold_search",
    "plot_basis_search",
    "plot_search",
    "scaled_epsilons",
    "validation_search",
]

DEFAULT_EPSILONS = np.logspace(-2, 1, 50)

_SCORES = ("gradient", "kfold")


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


@dataclass(frozen=True)
class BasisSearchResult:
    """Outcome of a joint search over the centre count and the shape parameter.

    The grid is rectangular and nothing is dropped from it — unlike :class:`SearchResult`,
    where badly conditioned candidates are filtered out of the returned arrays. In two
    dimensions it is a *cell* that gets rejected, not a whole row or column, so a filter
    cannot keep the grid rectangular. Rejected cells carry ``inf`` in ``scores`` instead.

    Attributes
    ----------
    best_n_centers : int
        The selected number of basis functions.
    best_epsilon : float
        The selected shape parameter.
    n_centers : ndarray of shape (m,)
        The centre counts tried, ascending.
    epsilons : ndarray of shape (p,)
        The shape parameters tried, ascending.
    scores : ndarray of shape (m, p)
        Error at each cell, indexed ``[centre count, shape parameter]``. Lower is better.
        ``inf`` marks a cell that was rejected for conditioning or failed to solve.
    conditions : ndarray of shape (m, p)
        Condition number of the full-data system at each cell. Unlike ``scores`` this
        reports the measured value even where the cell was rejected, so the conditioning
        landscape stays visible; only an outright solver failure records ``inf``.
    best_estimator : RBFRegressor or None
        A model fitted on all the data at the winning cell, unless ``refit=False``.
    rejected : int
        How many cells were excluded from the choice.
    n_samples : int
        Samples the search was run on, so the interpolation threshold can be marked
        without passing the data back in.
    """

    best_n_centers: int
    best_epsilon: float
    n_centers: NDArray[np.int_]
    epsilons: NDArray[np.float64]
    scores: NDArray[np.float64]
    conditions: NDArray[np.float64]
    best_estimator: RBFRegressor | None
    rejected: int
    n_samples: int

    def __repr__(self) -> str:
        return (
            f"BasisSearchResult(best_n_centers={self.best_n_centers}, "
            f"best_epsilon={self.best_epsilon:.4g}, "
            f"best_score={self.scores.min():.4g}, "
            f"grid={self.n_centers.size}x{self.epsilons.size}, "
            f"rejected={self.rejected}, n_samples={self.n_samples})"
        )


def kfold_search(
    estimator: RBFRegressor,
    X: ArrayLike,
    y: ArrayLike | None = None,
    dy: ArrayLike | None = None,
    *,
    k: int = 5,
    groups: ArrayLike | None = None,
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
    groups : array_like of shape (n_samples,), optional
        Group label per sample. When given, whole groups are held out together
        (:class:`~sklearn.model_selection.GroupKFold`) instead of individual rows.

        This matters whenever samples come in tightly spaced clusters, trajectory data
        being the clear case: a held-out row's two nearest neighbours are the points before
        and after it on its own trajectory, and both are still in the training set, so a
        row-wise fold error is close to an interpolation error and reports a model as far
        better than it is. Hold out whole trajectories and the number becomes honest. See
        :mod:`ge_rbf.trajectories`.
    epsilons : array_like, optional
        Candidate shape parameters. See :func:`scaled_epsilons` if the coordinates are not
        of order one.
    max_condition : float, optional
        Candidates whose full-data system is worse conditioned than this are rejected.
        Candidates whose system is singular to working precision are always rejected.
    refit : bool, optional
        Fit a model on all the data at the winning shape parameter and return it as
        ``best_estimator``.
    random_state : int or RandomState, optional
        Seeds the fold shuffle, so a search is reproducible. Ignored when ``groups`` is
        given, since the group split is determined by the labels.

    Returns
    -------
    SearchResult
    """
    X = check_samples(X)
    n_samples = X.shape[0]
    y_values, _ = check_targets(y, n_samples)
    gradients = None if dy is None else check_gradients(dy, n_samples, X.shape[1])

    folds = _folds(X, k, groups, random_state)

    def score(epsilon: float) -> tuple[float, None]:
        # None: the fold models say nothing about the conditioning of the full-data
        # system, so _search has to fit that separately.
        return _fold_error(estimator, epsilon, X, y_values, gradients, folds, {}), None

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

    def score(epsilon: float) -> tuple[float, float]:
        fitted = _fit(estimator, epsilon, X, y_values, gradients)
        return _relative_error(y_valid_values, fitted.predict(X_valid)), fitted.condition_

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

    def score(epsilon: float) -> tuple[float, float]:
        fitted = _fit(estimator, epsilon, X, y_values, fit_gradients)
        return _relative_error(gradients, fitted.predict_gradient(X)), fitted.condition_

    return _search(estimator, X, y_values, fit_gradients, epsilons, max_condition, refit, score)


def basis_search(
    estimator: RBFRegressor,
    X: ArrayLike,
    y: ArrayLike,
    dy: ArrayLike | None = None,
    *,
    n_centers: ArrayLike | None = None,
    epsilons: ArrayLike | None = None,
    score: str = "kfold",
    k: int = 5,
    groups: ArrayLike | None = None,
    max_condition: float = 1e13,
    refit: bool = True,
    random_state: int | np.random.Generator | None = 0,
) -> BasisSearchResult:
    """Search the number of basis functions and the shape parameter together.

    One centre per sample — the default — answers "how many basis functions?" by inheriting
    the answer from the sampling. When the samples are clustered that is the wrong answer:
    the centres end up packed together where the data is already dense and sparse where it
    is thin, neighbouring centres look nearly identical, and the system is ill conditioned
    by construction. Re-sampling the centres decouples the two, and the count then becomes
    a hyperparameter that has to be chosen. It interacts strongly with the shape parameter,
    so the two are searched jointly rather than one after the other.

    Parameters
    ----------
    estimator : RBFRegressor
        Template. It is cloned for every cell, never modified. Its ``centers`` and
        ``epsilon`` are overridden by the search.
    X, y : array_like
        Samples and function values.
    dy : array_like of shape (n_samples, n_features), optional
        Sampled gradients. Required by ``score="gradient"``; with ``score="kfold"`` they
        make the candidates gradient-enhanced.
    n_centers : array_like of int, optional
        Centre counts to try. ``None`` uses a geometric ladder from ``n_features + 1`` up
        to ``n_samples // 2`` — geometric because the interesting variation is at the low
        end, and stopping at half the samples because that keeps the default grid clear of
        the interpolation hazard described below.
    epsilons : array_like, optional
        Shape parameters to try. ``None`` uses ``scaled_epsilons(X)``, which is almost
        always what is wanted here: the centre count changes what a sensible width is, and
        an absolute range is only meaningful when the coordinates are of order one.
    score : {"kfold", "gradient"}, optional
        Where the error signal comes from.

        ``"kfold"`` (the default) holds out folds, using the same error expression as
        :func:`kfold_search`, and fits gradient-enhanced candidates when ``dy`` is given.
        Costs ``k + 1`` fits per cell. Pass ``groups`` with it whenever the samples are
        clustered.

        ``"gradient"`` fits each candidate to **function values only** and scores it
        against the sampled gradients, which are then never in the system being fitted.
        One fit per cell, so it is ``k + 1`` times cheaper and needs nothing held out —
        but it measured materially worse at choosing a centre count, see the notes.
    k : int, optional
        Number of folds, for ``score="kfold"``.
    groups : array_like of shape (n_samples,), optional
        Group labels, for ``score="kfold"``. Whole groups are held out together. Passing
        this with any other ``score`` raises rather than being silently ignored.
    max_condition : float, optional
        Cells whose full-data system is worse conditioned than this are excluded from the
        choice. A whole centre count going over the limit is information rather than an
        error: with the count free to vary there is usually a smaller one that is well
        conditioned, which is the point of searching it.
    refit : bool, optional
        Fit a model on all the data at the winning cell and return it as
        ``best_estimator``, using the same fitting mode the candidates used.
    random_state : int or Generator, optional
        Seeds the centre draw. Unlike most scikit-learn estimators this defaults to ``0``
        rather than ``None``, deliberately: with an unseeded draw every cell would get a
        different centre set and the grid would stop being a comparison.

    Returns
    -------
    BasisSearchResult

    Notes
    -----
    **Why the default holds data out, when the rest of the package does not have to.** The
    shape parameter alone can be chosen from the sampled gradients with nothing held out,
    which is what makes :func:`gradient_search` attractive. The centre count cannot, and
    the reason is structural: a gradient-enhanced model has the sampled gradients as rows
    of its own fitting system, so it reproduces them better the more centres it is given.
    That score falls with the count monotonically — measured across a full ladder it went
    from 8.2e-01 at 5 centres to 2.4e-03 at one per sample, never once turning back up —
    so it cannot choose a count at all, only the largest one on offer.

    Fitting function values only and scoring against gradients that were never in the
    system does fix the monotonicity: that curve does turn over. It is still the weaker
    signal. Measured on clustered trajectory data over five sample sets and two responses,
    judged by the error on whole designs no search had seen, the median was:

    ========================  ============  ============
    selector                  response A    response B
    ========================  ============  ============
    ``score="gradient"``            29.6%         47.4%
    ``score="kfold"``               16.4%         12.4%
    ``score="kfold"``, groups       15.5%         15.6%
    one centre per sample           49.8%         18.6%
    ========================  ============  ============

    So ``"kfold"`` is the default, and ``"gradient"`` stays available as the cheap option
    with its cost stated. The last row is the baseline this function exists to beat.

    **Pass ``groups`` when the samples are clustered.** In the same experiment, row-wise
    folds selected the largest count in the grid on all ten runs, while group-wise folds
    chose interior counts and were far steadier across sample sets. Row-wise folds hold out
    a point whose nearest neighbours are still in the training set, which under-penalises a
    large basis; the median above hides that because saturating at a ladder that stops at
    ``n_samples // 2`` is a reasonable place to end up by accident.

    To obtain a gradient-enhanced model at the chosen setting::

        model = clone(estimator).set_params(
            centers=result.best_n_centers, epsilon=result.best_epsilon
        ).fit(X, y, dy=dy)

    **Do not report accuracy on data the search consumed.** The winning cell was chosen by
    looking at these samples, so an error measured on them is optimistic by however much
    that choice bought. Keep a set of held-out samples — whole groups, for trajectory data
    — that no search has seen.

    Examples
    --------
    >>> from ge_rbf import RBFRegressor, TrajectoryScaler
    >>> from ge_rbf.problems import load_path, load_path_samples
    >>> from ge_rbf.selection import basis_search
    >>> Z, groups = load_path_samples()
    >>> y, dy = load_path(Z)
    >>> scaler = TrajectoryScaler().fit(Z, groups=groups)
    >>> result = basis_search(
    ...     RBFRegressor(), scaler.transform(Z), y, scaler.transform_gradient(dy),
    ...     groups=groups,
    ... )
    >>> result.best_n_centers in result.n_centers
    True
    """
    X = check_samples(X)
    n_samples, n_features = X.shape
    y_values, _ = check_targets(y, n_samples)
    gradients = None if dy is None else check_gradients(dy, n_samples, n_features)

    if score not in _SCORES:
        raise ValueError(f"score must be one of {_SCORES}, got {score!r}.")

    if groups is not None and score != "kfold":
        raise ValueError(
            f"groups only applies to score='kfold', but score={score!r}. Held-out groups "
            "have no meaning for a score that holds nothing out."
        )

    if score == "gradient" and gradients is None:
        raise ValueError("score='gradient' scores against the sampled gradients; pass dy.")

    counts = _center_counts(n_centers, n_samples, n_features)
    candidates = scaled_epsilons(X) if epsilons is None else np.asarray(epsilons, np.float64)
    candidates = np.sort(candidates.ravel())

    if candidates.size == 0:
        raise ValueError("epsilons must contain at least one candidate.")
    if np.any(candidates <= 0):
        raise ValueError("epsilons must all be positive.")
    if max_condition <= 0:
        raise ValueError(f"max_condition must be positive, got {max_condition}.")

    if score == "gradient" and np.any(counts >= n_samples):
        warnings.warn(
            f"{int(np.sum(counts >= n_samples))} of the requested centre counts are at or "
            f"above n_samples={n_samples}. score='gradient' holds nothing out, so the "
            "score falls with the centre count by construction there and the search will "
            "tend to prefer those cells even though they generalise worse. Use "
            "score='kfold' with groups, or keep the counts below n_samples.",
            RuntimeWarning,
            stacklevel=2,
        )

    # "gradient" scores against gradients that were never in the system, so candidates are
    # fitted to function values alone. "kfold" holds data out, so it can use them in the fit.
    fit_gradients = None if score == "gradient" else gradients
    folds = _folds(X, k, groups, random_state) if score == "kfold" else []

    scores = np.full((counts.size, candidates.size), np.inf)
    conditions = np.full((counts.size, candidates.size), np.inf)

    for i, count in enumerate(counts):
        params = {"centers": int(count), "random_state": random_state}

        for j, epsilon in enumerate(candidates):
            try:
                if score == "gradient":
                    fitted = _fit(estimator, epsilon, X, y_values, None, **params)
                    error = _relative_error(gradients, fitted.predict_gradient(X))
                    condition = fitted.condition_
                else:
                    error = _fold_error(estimator, epsilon, X, y_values, gradients, folds, params)
                    condition = _fit(
                        estimator, epsilon, X, y_values, fit_gradients, **params
                    ).condition_
            except np.linalg.LinAlgError:
                # Singular to working precision: a cell to discard, not an error to
                # propagate out of a sweep.
                continue

            conditions[i, j] = condition
            if np.isfinite(error) and np.isfinite(condition) and condition <= max_condition:
                scores[i, j] = error

    rejected = int(np.sum(~np.isfinite(scores)))

    if not np.any(np.isfinite(scores)):
        raise ValueError(
            f"Every cell was rejected: none produced a system better conditioned than "
            f"max_condition={max_condition:.3g} (best was {np.nanmin(conditions):.3g}). "
            f"Raise max_condition, lower the centre counts, or try larger shape parameters "
            f"— wide basis functions are what make the system ill-conditioned."
        )

    best_i, best_j = np.unravel_index(np.argmin(scores), scores.shape)
    best_n_centers = int(counts[best_i])
    best_epsilon = float(candidates[best_j])

    best_estimator = None
    if refit:
        best_estimator = _fit(
            estimator,
            best_epsilon,
            X,
            y_values,
            fit_gradients,
            centers=best_n_centers,
            random_state=random_state,
        )

    return BasisSearchResult(
        best_n_centers=best_n_centers,
        best_epsilon=best_epsilon,
        n_centers=counts,
        epsilons=candidates,
        scores=scores,
        conditions=conditions,
        best_estimator=best_estimator,
        rejected=rejected,
        n_samples=n_samples,
    )


def scaled_epsilons(X: ArrayLike, epsilons: ArrayLike = DEFAULT_EPSILONS) -> NDArray[np.float64]:
    r"""Candidate shape parameters expressed relative to the spacing of the samples.

    The Gaussian kernel is :math:`\exp(-\varepsilon \|x - c\|^2)`, so :math:`\varepsilon`
    multiplies the **squared** distance. A candidate set that means the same thing however
    the coordinates are scaled therefore divides by the square of a length:

    .. math:: \varepsilon \;\rightarrow\; \varepsilon / h^2

    with :math:`h` the median distance between distinct samples. Dividing by :math:`h`
    instead is a natural-looking mistake that leaves the candidates moving in the right
    direction but landing in the wrong place, by a factor that grows the further :math:`h`
    sits from one.

    Use this whenever the coordinates are not of order one — in particular after a
    coordinate frame, which stretches each axis by its curvature and so can move the
    sample spacing by orders of magnitude. With absolute candidates the whole sweep then
    lands outside the useful range, and the failure is quiet rather than loud: every basis
    function underflows, the fitted weights are meaningless, and the model predicts a flat
    zero without raising anything.

    Parameters
    ----------
    X : array_like of shape (n_samples, n_features)
        The samples the model will be fitted to, **in the coordinates it will be fitted
        in**.
    epsilons : array_like, optional
        Candidates in units of inverse squared spacing. The default sweeps the same three
        decades as :data:`DEFAULT_EPSILONS`.

    Returns
    -------
    ndarray
        Shape parameters ready to pass to any of the searches.

    Examples
    --------
    >>> import numpy as np
    >>> from ge_rbf.selection import scaled_epsilons
    >>> X = np.linspace(0, 100, 20).reshape(-1, 1)
    >>> candidates = scaled_epsilons(X)
    >>> float(candidates.min()) < 1e-3
    True
    """
    X = check_samples(X)
    if X.shape[0] < 2:
        raise ValueError("At least 2 samples are needed to measure a spacing.")

    spacing = _median_spacing(X)
    epsilons = np.asarray(epsilons, dtype=np.float64)

    return epsilons / spacing**2


def _median_spacing(X: NDArray[np.float64]) -> float:
    """Median distance between distinct samples: the length scale the basis must match."""
    spacing = float(np.median(pdist(X)))

    if spacing <= 0:
        raise ValueError(
            "The median distance between samples is zero, so there is no length scale to "
            "measure the shape parameter against. More than half the samples coincide."
        )

    return spacing


# --------------------------------------------------------------------------- internals


def _center_counts(
    n_centers: ArrayLike | None, n_samples: int, n_features: int
) -> NDArray[np.int_]:
    """Validate the centre-count grid, or build the default geometric ladder."""
    if n_centers is None:
        lowest = max(4, n_features + 1)
        highest = max(lowest + 1, n_samples // 2)
        ladder = np.geomspace(lowest, highest, 8)
        return np.unique(np.round(ladder).astype(int))

    counts = np.asarray(n_centers).ravel()
    if counts.size == 0:
        raise ValueError("n_centers must contain at least one count.")
    if not np.all(counts == np.round(counts)):
        raise ValueError("n_centers must all be whole numbers.")

    counts = np.unique(counts.astype(int))
    if np.any(counts < 1):
        raise ValueError(f"n_centers must all be at least 1, got a minimum of {counts.min()}.")

    return counts


def _fold_error(
    estimator: RBFRegressor,
    epsilon: float,
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    gradients: NDArray[np.float64] | None,
    folds: list[tuple[NDArray[np.int_], NDArray[np.int_]]],
    params: dict[str, object],
) -> float:
    """Mean held-out error over the folds, matching :func:`kfold_search`'s expression."""
    errors = []

    for train, validate in folds:
        train_gradients = None if gradients is None else gradients[train]
        fitted = _fit(estimator, epsilon, X[train], y[train], train_gradients, **params)
        predicted = fitted.predict(X[validate])

        if gradients is None:
            errors.append(_rmse(y[validate], predicted))
        else:
            # Put the two error terms on a comparable footing before adding them.
            balance = np.mean(np.abs(gradients[train])) / np.mean(np.abs(y[train]))
            errors.append(
                balance * _rmse(y[validate], predicted)
                + _rmse(gradients[validate], fitted.predict_gradient(X[validate]))
            )

    return float(np.mean(errors))


def _folds(
    X: NDArray[np.float64],
    k: int,
    groups: ArrayLike | None,
    random_state: int | np.random.RandomState | None,
) -> list[tuple[NDArray[np.int_], NDArray[np.int_]]]:
    """Train/validate index pairs, holding out whole groups when labels are supplied."""
    n_samples = X.shape[0]

    if groups is None:
        if not 2 <= k <= n_samples:
            raise ValueError(f"k must be between 2 and n_samples={n_samples}, got {k}.")
        return list(KFold(n_splits=k, shuffle=True, random_state=random_state).split(X))

    labels = np.asarray(groups).ravel()
    if labels.shape[0] != n_samples:
        raise ValueError(f"groups has {labels.shape[0]} entries but there are {n_samples} samples.")

    n_groups = np.unique(labels).size
    if not 2 <= k <= n_groups:
        raise ValueError(
            f"k must be between 2 and the number of groups={n_groups} when groups is "
            f"given, got {k}."
        )

    return list(GroupKFold(n_splits=k).split(X, groups=labels))


def _fit(
    estimator: RBFRegressor,
    epsilon: float,
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    gradients: NDArray[np.float64] | None,
    **params: object,
) -> RBFRegressor:
    """Clone the template at ``epsilon`` and fit it to the given data.

    Cloning is what keeps a search side-effect free: the caller's estimator is a template
    that is never fitted or reparameterised. Any extra ``params`` are set on the clone
    alongside the shape parameter, which is how :func:`basis_search` varies the centre
    count.

    When ``centers`` is left at its default, a model fitted on a fold is centred on that
    fold's own training points, matching how the folds were originally handled. An
    explicit centre set is shared across folds unchanged.

    When ``centers`` is an integer the centres are re-drawn on each call, over the bounding
    box of whatever data that call was given — on a fold, that fold's own box. This is the
    same rule as the default: a fold model is fitted only on its training set, including
    where its basis lives. Sharing one centre set across every fold would instead leak the
    extent of the held-out data into each fold model, which matters exactly when whole
    groups are held out. A fixed seed keeps the underlying unit hypercube identical across
    calls, so the only thing that moves between folds is the box.
    """
    candidate = clone(estimator)
    candidate.set_params(epsilon=float(epsilon), **params)
    return candidate.fit(X, y, dy=gradients)


def _search(
    estimator: RBFRegressor,
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    gradients: NDArray[np.float64] | None,
    epsilons: ArrayLike,
    max_condition: float,
    refit: bool,
    score: Callable[[float], tuple[float, float | None]],
) -> SearchResult:
    """Score every candidate, drop the badly conditioned ones, and pick the best.

    ``score`` returns ``(error, condition)``. A search whose scoring already fits the
    full-data model reports its condition number from that same fit; ``None`` means the
    condition has to be measured with an extra fit, which is the case for k-fold, where
    scoring only ever fits fold models.
    """
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
        try:
            scores[i], condition = score(epsilon)
            if condition is None:
                condition = _fit(estimator, epsilon, X, y, gradients).condition_
            conditions[i] = condition
        except np.linalg.LinAlgError:
            # Wide basis functions make every centre look alike, and past some point the
            # system is singular to working precision. That is a candidate to discard, not
            # an error to propagate out of a sweep.
            conditions[i] = np.inf
            scores[i] = np.inf

    # Reject ill-conditioned candidates *before* choosing the winner. The previous
    # implementation applied this filter only when drawing the diagnostic plot, so
    # max_condition had no effect on the value it returned.
    keep = np.isfinite(conditions) & (conditions <= max_condition)
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


def plot_basis_search(
    result: BasisSearchResult, ax: Axes | None = None, values: str = "score"
) -> Axes:
    """Plot a joint search as a heatmap over centre count and shape parameter.

    A heatmap rather than one line per centre count, because the thing worth seeing is
    that the two axes trade off against each other — the best shape parameter moves as the
    count changes — and a bundle of lines hides exactly that. ``values="condition"`` draws
    the conditioning over the same grid, which is the other half of the picture: the useful
    region is where the score is low *and* the system is still solvable.

    Parameters
    ----------
    result : BasisSearchResult
    ax : matplotlib Axes, optional
        Draw on these axes instead of creating a new figure.
    values : {"score", "condition"}, optional
        Which grid to draw.

    Returns
    -------
    matplotlib Axes
    """
    import matplotlib.pyplot as plt

    if values not in ("score", "condition"):
        raise ValueError(f"values must be 'score' or 'condition', got {values!r}.")

    if ax is None:
        _, ax = plt.subplots()

    grid = result.scores if values == "score" else result.conditions
    with np.errstate(divide="ignore"):
        shaded = np.ma.masked_invalid(np.log10(grid))

    mesh = ax.pcolormesh(result.epsilons, result.n_centers, shaded, shading="nearest")
    ax.figure.colorbar(mesh, ax=ax, label=f"log10({values})")

    ax.plot(result.best_epsilon, result.best_n_centers, "k.", markersize=14, label="Selected")

    # Where the function-value system turns square, i.e. where the fit stops being a
    # regression and starts interpolating.
    if result.n_centers.min() <= result.n_samples <= result.n_centers.max():
        ax.axhline(result.n_samples, color="k", linestyle="--", linewidth=1, label="Interpolation")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Shape parameter")
    ax.set_ylabel("Number of centres")
    ax.set_title(f"Search {values} vs basis size and shape parameter")
    ax.legend()

    return ax
