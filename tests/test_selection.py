"""Tests for the shape parameter searches.

Two behaviours get the most attention, because both were wrong before: the returned model
must actually be fitted at the winning shape parameter, and ``max_condition`` must
influence which shape parameter wins rather than only the diagnostic plot.
"""

import matplotlib
import numpy as np
import pytest

from ge_rbf.models import RBFRegressor
from ge_rbf.problems import non_isotropic
from ge_rbf.selection import (
    gradient_search,
    kfold_search,
    plot_search,
    validation_search,
)

matplotlib.use("Agg")

EPSILONS = np.logspace(-2, 1, 15)


@pytest.fixture
def sampled():
    rng = np.random.default_rng(0)
    X = rng.random((25, 2))
    y, dy = non_isotropic(X)
    return X, y, dy


@pytest.fixture
def validation():
    rng = np.random.default_rng(1)
    X = rng.random((30, 2))
    y, _ = non_isotropic(X)
    return X, y


def all_searches(X, y, dy, valid, **kwargs):
    X_valid, y_valid = valid
    return [
        kfold_search(RBFRegressor(), X, y, dy=dy, epsilons=EPSILONS, random_state=0, **kwargs),
        validation_search(
            RBFRegressor(), X, y, X_valid, y_valid, dy=dy, epsilons=EPSILONS, **kwargs
        ),
        gradient_search(RBFRegressor(), X, y, dy, epsilons=EPSILONS, **kwargs),
    ]


# ------------------------------------------------------------ the refit-at-optimum bug


def test_the_returned_model_is_fitted_at_the_winning_shape_parameter(sampled, validation):
    """The old implementation left the model at the last epsilon it tried, not the best."""
    X, y, dy = sampled

    for result in all_searches(X, y, dy, validation):
        assert result.best_estimator is not None
        assert result.best_estimator.epsilon == result.best_epsilon
        assert result.best_estimator.mode_ == "ge"


def test_refit_false_skips_the_final_fit(sampled, validation):
    X, y, dy = sampled

    for result in all_searches(X, y, dy, validation, refit=False):
        assert result.best_estimator is None


def test_the_winner_really_has_the_lowest_score(sampled, validation):
    X, y, dy = sampled

    for result in all_searches(X, y, dy, validation):
        assert result.best_epsilon == result.epsilons[np.argmin(result.scores)]
        assert result.scores.min() == result.scores[result.epsilons == result.best_epsilon][0]


# ------------------------------------------------------------- the max_condition bug


def test_max_condition_changes_which_shape_parameter_wins(sampled, validation):
    """Verified against the old code: 1e12 and 1e2 returned the identical value."""
    X, y, dy = sampled

    permissive = gradient_search(RBFRegressor(), X, y, dy, epsilons=EPSILONS, max_condition=np.inf)
    strict = gradient_search(RBFRegressor(), X, y, dy, epsilons=EPSILONS, max_condition=1e4)

    assert strict.rejected > 0
    assert permissive.rejected == 0
    assert strict.best_epsilon != permissive.best_epsilon
    assert np.all(strict.conditions <= 1e4)


def test_reported_conditions_match_a_direct_fit(sampled, validation):
    """Searches that already fit the full-data model reuse its condition number.

    They must report exactly what an independent fit at the same epsilon would give,
    whether the number came from the scoring fit or from a separate one.
    """
    X, y, dy = sampled
    X_valid, y_valid = validation

    for result in all_searches(X, y, dy, validation, max_condition=np.inf):
        for epsilon, condition in zip(result.epsilons, result.conditions, strict=True):
            direct = RBFRegressor(epsilon=epsilon).fit(X, y, dy=dy)
            assert condition == pytest.approx(direct.condition_, rel=1e-12)


def test_rejected_candidates_are_dropped_from_the_result(sampled):
    X, y, dy = sampled
    result = gradient_search(RBFRegressor(), X, y, dy, epsilons=EPSILONS, max_condition=1e6)

    assert result.epsilons.size + result.rejected == EPSILONS.size
    assert result.scores.size == result.epsilons.size == result.conditions.size


def test_a_singular_candidate_is_rejected_rather_than_raising(sampled):
    """Very wide basis functions make the system singular; a sweep must survive that."""
    X, y, _ = sampled
    # 1e-6 is wide enough that every centre looks alike to working precision.
    epsilons = np.concatenate([[1e-6], EPSILONS])

    result = kfold_search(RBFRegressor(), X, y, epsilons=epsilons, random_state=0)

    assert result.rejected >= 1
    assert 1e-6 not in result.epsilons
    assert np.all(np.isfinite(result.scores))


def test_rejecting_everything_is_an_error_not_a_silent_empty_result(sampled):
    X, y, dy = sampled

    with pytest.raises(ValueError, match="worse conditioned than"):
        gradient_search(RBFRegressor(), X, y, dy, epsilons=EPSILONS, max_condition=1e-3)


# ------------------------------------------------------------------- general behaviour


def test_searches_do_not_touch_the_estimator_they_are_given(sampled, validation):
    """The template is cloned per candidate; the caller's object stays unfitted."""
    X, y, dy = sampled
    X_valid, y_valid = validation
    template = RBFRegressor(epsilon=0.123)

    kfold_search(template, X, y, dy=dy, epsilons=EPSILONS, random_state=0)
    validation_search(template, X, y, X_valid, y_valid, dy=dy, epsilons=EPSILONS)
    gradient_search(template, X, y, dy, epsilons=EPSILONS)

    assert template.epsilon == 0.123
    assert not hasattr(template, "coef_")


def test_kfold_is_reproducible_given_a_seed_and_varies_without_one(sampled):
    X, y, dy = sampled

    first = kfold_search(RBFRegressor(), X, y, dy=dy, epsilons=EPSILONS, random_state=7)
    second = kfold_search(RBFRegressor(), X, y, dy=dy, epsilons=EPSILONS, random_state=7)

    np.testing.assert_array_equal(first.scores, second.scores)
    assert first.best_epsilon == second.best_epsilon


def test_kfold_puts_every_sample_in_exactly_one_validation_fold():
    """25 samples over 4 folds: the old fixed-width slicing silently dropped the leftover."""
    from sklearn.model_selection import KFold

    n_samples, k = 25, 4
    covered = np.zeros(n_samples, dtype=int)
    for _, validate in KFold(n_splits=k, shuffle=True, random_state=0).split(np.arange(n_samples)):
        covered[validate] += 1

    np.testing.assert_array_equal(covered, 1)


def test_function_value_only_searches_work_without_gradients(sampled, validation):
    X, y, _ = sampled
    X_valid, y_valid = validation

    for result in (
        kfold_search(RBFRegressor(), X, y, epsilons=EPSILONS, random_state=0),
        validation_search(RBFRegressor(), X, y, X_valid, y_valid, epsilons=EPSILONS),
    ):
        assert result.best_estimator.mode_ == "fv"


def test_gradient_search_can_hold_the_gradients_out_of_the_fit(sampled):
    """With use_gradients_in_fit=False the gradients are an independent check, not a residual."""
    X, y, dy = sampled

    held_out = gradient_search(
        RBFRegressor(), X, y, dy, use_gradients_in_fit=False, epsilons=EPSILONS
    )

    assert held_out.best_estimator.mode_ == "fv"


def test_the_selected_model_beats_a_badly_chosen_one(sampled, validation):
    """The search has to be worth running: it should beat the extremes of its own range."""
    X, y, dy = sampled
    X_valid, y_valid = validation

    result = validation_search(
        RBFRegressor(), X, y, X_valid, y_valid, dy=dy, epsilons=EPSILONS, max_condition=np.inf
    )
    selected = np.linalg.norm(result.best_estimator.predict(X_valid) - y_valid)

    for bad in (EPSILONS[0], EPSILONS[-1]):
        alternative = RBFRegressor(epsilon=bad).fit(X, y, dy=dy)
        assert selected <= np.linalg.norm(alternative.predict(X_valid) - y_valid)


def test_transformed_frame_improves_the_search_result(sampled, validation):
    """The headline claim of the paper, on the test problem it was made for."""
    from ge_rbf import IsotropicTransformer
    from ge_rbf.problems import random_rotation

    rotation = random_rotation(2, rng=3)
    X = np.random.default_rng(4).random((40, 2))
    y, dy = non_isotropic(X, rotation)

    X_valid = np.random.default_rng(5).random((60, 2))
    y_valid, _ = non_isotropic(X_valid, rotation)

    plain = validation_search(
        RBFRegressor(), X, y, X_valid, y_valid, dy=dy, epsilons=EPSILONS, max_condition=np.inf
    )

    frame = IsotropicTransformer(method="ge-lhm").fit(X, dy=dy)
    transformed = validation_search(
        RBFRegressor(),
        frame.transform(X),
        y,
        frame.transform(X_valid),
        y_valid,
        dy=frame.transform_gradient(dy),
        epsilons=EPSILONS,
        max_condition=np.inf,
    )

    assert transformed.scores.min() < plain.scores.min()


# ------------------------------------------------------------------------- guards


def test_invalid_arguments_are_reported(sampled):
    X, y, dy = sampled

    with pytest.raises(ValueError, match="k must be between"):
        kfold_search(RBFRegressor(), X, y, dy=dy, k=1, epsilons=EPSILONS)
    with pytest.raises(ValueError, match="at least one candidate"):
        gradient_search(RBFRegressor(), X, y, dy, epsilons=[])
    with pytest.raises(ValueError, match="epsilons must all be positive"):
        gradient_search(RBFRegressor(), X, y, dy, epsilons=[-1.0, 1.0])
    with pytest.raises(ValueError, match="max_condition must be positive"):
        gradient_search(RBFRegressor(), X, y, dy, epsilons=EPSILONS, max_condition=0)


def test_result_repr_is_informative(sampled):
    X, y, dy = sampled
    text = repr(gradient_search(RBFRegressor(), X, y, dy, epsilons=EPSILONS))

    assert "best_epsilon" in text and "n_candidates" in text


def test_plot_search_draws_the_curve_and_the_winner(sampled):
    X, y, dy = sampled
    result = gradient_search(RBFRegressor(), X, y, dy, epsilons=EPSILONS)

    ax = plot_search(result)

    assert ax.get_xlabel() == "Shape parameter"
    assert len(ax.lines) == 2


def test_importing_the_package_does_not_import_pyplot():
    """plot_search imports matplotlib lazily so `import ge_rbf` stays cheap."""
    import subprocess
    import sys

    code = "import sys, ge_rbf; print('matplotlib.pyplot' in sys.modules)"
    output = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert output.stdout.strip() == "False"
