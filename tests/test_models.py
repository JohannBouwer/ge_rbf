"""Tests for :class:`ge_rbf.RBFRegressor`."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from ge_rbf.models import RBFRegressor
from ge_rbf.problems import non_isotropic


@pytest.fixture
def sampled():
    rng = np.random.default_rng(0)
    X = rng.random((20, 2))
    y, dy = non_isotropic(X)
    return X, y, dy


# ------------------------------------------------------------------ mode inference


def test_mode_is_inferred_from_the_arguments(sampled):
    X, y, dy = sampled

    assert RBFRegressor().fit(X, y).mode_ == "fv"
    assert RBFRegressor().fit(X, y, dy=dy).mode_ == "ge"
    assert RBFRegressor().fit(X, dy=dy, anchor_X=X[:1], anchor_y=y[:1]).mode_ == "go"


def test_ambiguous_or_incomplete_calls_are_rejected(sampled):
    X, y, dy = sampled

    with pytest.raises(ValueError, match="y is required"):
        RBFRegressor().fit(X)
    with pytest.raises(ValueError, match="must be supplied together"):
        RBFRegressor().fit(X, dy=dy, anchor_X=X[:1])
    with pytest.raises(ValueError, match="needs gradients"):
        RBFRegressor().fit(X, anchor_X=X[:1], anchor_y=y[:1])
    with pytest.raises(ValueError, match="not both"):
        RBFRegressor().fit(X, y, dy=dy, anchor_X=X[:1], anchor_y=y[:1])


# ------------------------------------------------------------------ numerical behaviour


def test_function_value_model_interpolates_its_training_data(sampled):
    X, y, _ = sampled
    model = RBFRegressor(epsilon=1.0).fit(X, y)

    np.testing.assert_allclose(model.predict(X), y, atol=1e-9)


def test_gradient_enhanced_model_reproduces_sampled_gradients(sampled):
    """A GE fit is a regression, so it trades off both blocks rather than interpolating."""
    X, y, dy = sampled
    model = RBFRegressor(epsilon=0.5).fit(X, y, dy=dy)

    predicted = model.predict_gradient(X)
    relative_error = np.linalg.norm(predicted - dy) / np.linalg.norm(dy)
    assert relative_error < 0.05


def test_predicted_gradient_matches_finite_differences_of_the_prediction(sampled):
    """The analytic gradient of the surrogate must be the gradient of the surrogate."""
    X, y, dy = sampled
    model = RBFRegressor(epsilon=0.9).fit(X, y, dy=dy)

    probe = np.random.default_rng(1).random((8, 2))
    analytic = model.predict_gradient(probe)

    # A relatively wide step: the fitted weights reach ~1e6 and largely cancel, so a
    # smaller step is dominated by round-off in the differenced predictions rather than by
    # truncation error.
    step = 1e-4
    numeric = np.empty_like(probe)
    for f in range(probe.shape[1]):
        forward, backward = probe.copy(), probe.copy()
        forward[:, f] += step
        backward[:, f] -= step
        numeric[:, f] = (model.predict(forward) - model.predict(backward)) / (2 * step)

    np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-6)


def test_gradient_only_model_recovers_gradients_but_needs_the_anchor_for_the_offset(sampled):
    X, y, dy = sampled
    model = RBFRegressor(epsilon=0.5).fit(X, dy=dy, anchor_X=X[:1], anchor_y=y[:1])

    gradient_error = np.linalg.norm(model.predict_gradient(X) - dy) / np.linalg.norm(dy)
    assert gradient_error < 0.05

    # The anchor pins the constant offset. It is one row of a least-squares system rather
    # than a hard constraint, so it is reproduced closely but not exactly.
    np.testing.assert_allclose(model.predict(X[:1]), y[:1], rtol=1e-3)


def test_gradient_only_model_without_a_matching_anchor_is_offset(sampled):
    """Gradients alone fix the surrogate only up to a constant; the anchor supplies it."""
    X, y, dy = sampled
    shifted = RBFRegressor(epsilon=0.5).fit(X, dy=dy, anchor_X=X[:1], anchor_y=y[:1] + 10.0)
    matched = RBFRegressor(epsilon=0.5).fit(X, dy=dy, anchor_X=X[:1], anchor_y=y[:1])

    # Both recover essentially the same gradient field. Not exactly: a sum of Gaussians
    # cannot represent a constant, so absorbing the offset perturbs the weights slightly.
    np.testing.assert_allclose(shifted.predict_gradient(X), matched.predict_gradient(X), rtol=2e-2)
    # ...but the anchored function values differ by the offset that was imposed.
    assert shifted.predict(X[:1]) - matched.predict(X[:1]) > 9.0


def test_fewer_centres_than_samples_gives_a_regression_fit(sampled):
    """The old implementation raised LinAlgError here because it always used a direct solve."""
    X, y, _ = sampled
    centers = X[::3]
    model = RBFRegressor(epsilon=1.0, centers=centers).fit(X, y)

    assert model.coef_.shape == (centers.shape[0],)

    # The weights are the least-squares solution of the (over-determined) system.
    from ge_rbf.kernels import GaussianKernel

    expected, *_ = np.linalg.lstsq(GaussianKernel(1.0)(X, centers), y, rcond=None)
    np.testing.assert_allclose(model.coef_, expected, rtol=1e-10)


def test_more_centres_fit_the_data_better(sampled):
    """Sanity check on the regression path: added flexibility must reduce the residual."""
    X, y, _ = sampled

    residuals = [
        np.linalg.norm(RBFRegressor(epsilon=1.0, centers=X[::step]).fit(X, y).predict(X) - y)
        for step in (4, 2, 1)
    ]

    assert residuals[0] > residuals[1] > residuals[2]


def test_more_centres_than_samples_is_allowed(sampled):
    X, y, _ = sampled
    extra = np.vstack([X, np.random.default_rng(2).random((5, 2))])
    model = RBFRegressor(epsilon=1.0, centers=extra).fit(X, y)

    assert model.coef_.shape == (extra.shape[0],)


def test_condition_number_is_recorded_in_every_mode(sampled):
    X, y, dy = sampled

    for kwargs in (
        {"y": y},
        {"y": y, "dy": dy},
        {"dy": dy, "anchor_X": X[:1], "anchor_y": y[:1]},
    ):
        model = RBFRegressor(epsilon=0.5).fit(X, **kwargs)
        assert np.isfinite(model.condition_) and model.condition_ > 0


def test_condition_number_grows_as_the_basis_functions_widen(sampled):
    X, y, _ = sampled
    wide = RBFRegressor(epsilon=0.01).fit(X, y).condition_
    narrow = RBFRegressor(epsilon=10.0).fit(X, y).condition_
    assert wide > narrow


# ------------------------------------------------------------------ shapes and API


@pytest.mark.parametrize("column", [False, True])
def test_target_shape_round_trips(sampled, column):
    X, y, _ = sampled
    target = y.reshape(-1, 1) if column else y

    prediction = RBFRegressor(epsilon=1.0).fit(X, target).predict(X)

    assert prediction.shape == target.shape


def test_single_feature_input_is_accepted_as_1d():
    X = np.linspace(0, 1, 9)
    y = np.sin(3 * np.pi * X)
    dy = (3 * np.pi * np.cos(3 * np.pi * X)).reshape(-1, 1)

    model = RBFRegressor(epsilon=2.0).fit(X, y, dy=dy)

    assert model.n_features_in_ == 1
    assert model.predict(X).shape == (9,)
    assert model.predict_gradient(X).shape == (9, 1)


def test_gradient_weight_auto_balances_the_two_blocks(sampled):
    """With gradients an order of magnitude larger, the weighting should change the fit."""
    X, y, dy = sampled

    unweighted = RBFRegressor(epsilon=0.5).fit(X, y, dy=dy)
    weighted = RBFRegressor(epsilon=0.5, gradient_weight="auto").fit(X, y, dy=dy)

    assert not np.allclose(unweighted.coef_, weighted.coef_)
    # Down-weighting gradients must improve the function-value fit.
    assert np.linalg.norm(weighted.predict(X) - y) < np.linalg.norm(unweighted.predict(X) - y)


def test_gradient_weight_of_one_is_the_default_behaviour(sampled):
    X, y, dy = sampled
    default = RBFRegressor(epsilon=0.5).fit(X, y, dy=dy)
    explicit = RBFRegressor(epsilon=0.5, gradient_weight=1.0).fit(X, y, dy=dy)

    np.testing.assert_allclose(default.coef_, explicit.coef_, rtol=1e-12)


def test_invalid_parameters_are_rejected(sampled):
    X, y, dy = sampled

    with pytest.raises(ValueError, match="solver must be one of"):
        RBFRegressor(solver="magic").fit(X, y)
    with pytest.raises(ValueError, match="gradient_weight must be positive"):
        RBFRegressor(gradient_weight=-1.0).fit(X, y, dy=dy)
    with pytest.raises(ValueError, match="needs a square system"):
        RBFRegressor(solver="solve").fit(X, y, dy=dy)


def test_mismatched_feature_counts_are_rejected(sampled):
    X, y, _ = sampled

    with pytest.raises(ValueError, match="centers have 3 features"):
        RBFRegressor(centers=np.zeros((4, 3))).fit(X, y)

    model = RBFRegressor().fit(X, y)
    with pytest.raises(ValueError, match="X has 3 features"):
        model.predict(np.zeros((4, 3)))


def test_predicting_before_fitting_raises():
    with pytest.raises(NotFittedError):
        RBFRegressor().predict(np.zeros((2, 2)))


def test_is_a_well_behaved_sklearn_estimator(sampled):
    X, y, _ = sampled
    model = RBFRegressor(epsilon=0.5, gradient_weight="auto")

    assert model.get_params()["epsilon"] == 0.5
    assert clone(model).get_params() == model.get_params()

    model.set_params(epsilon=2.0)
    assert model.epsilon == 2.0

    # RegressorMixin.score works, which is what makes GridSearchCV usable for FV models.
    assert model.fit(X, y).score(X, y) > 0.99
