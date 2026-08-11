"""Tests for :class:`ge_rbf.IsotropicTransformer`.

Two families of test matter here. First, does the curvature estimate recover a curvature
we know? A quadratic has a constant, exactly known Hessian, so every method can be checked
against ground truth rather than against itself. Second, is the frame applied consistently
— round trips, and the gradient chain rule against finite differences.
"""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from ge_rbf.problems import non_isotropic, random_rotation
from ge_rbf.transformations import IsotropicTransformer


def quadratic(X, hessian):
    """f(x) = 0.5 x' H x, with its exact gradient. Hessian is H everywhere."""
    f = 0.5 * np.einsum("ij,jk,ik->i", X, hessian, X)
    return f, X @ hessian


@pytest.fixture
def anisotropic_quadratic():
    """A quadratic whose curvature is 100x stronger in one direction than the other."""
    hessian = np.diag([100.0, 1.0])
    X = np.random.default_rng(0).random((30, 2)) * 2 - 1
    y, dy = quadratic(X, hessian)
    return X, y, dy, hessian


@pytest.fixture
def sampled():
    rng = np.random.default_rng(1)
    X = rng.random((40, 3))
    y, dy = non_isotropic(X)
    return X, y, dy


# ------------------------------------------------------- recovering known curvature


def test_ge_lhm_recovers_a_known_diagonal_hessian(anisotropic_quadratic):
    X, y, dy, hessian = anisotropic_quadratic

    frame = IsotropicTransformer(method="ge-lhm").fit(X, dy=dy)

    np.testing.assert_allclose(frame.curvature_, hessian, rtol=1e-8)
    np.testing.assert_allclose(frame.eigenvalues_, [100.0, 1.0], rtol=1e-8)
    np.testing.assert_allclose(frame.scaling_, [10.0, 1.0], rtol=1e-8)


def test_fv_lhm_recovers_a_known_diagonal_hessian(anisotropic_quadratic):
    X, y, _, hessian = anisotropic_quadratic

    frame = IsotropicTransformer(method="fv-lhm").fit(X, y=y)

    np.testing.assert_allclose(frame.curvature_, hessian, rtol=1e-6, atol=1e-8)


def test_ge_lhm_recovers_a_known_rotation():
    """With a rotated quadratic, the estimated frame must diagonalise the true Hessian."""
    rotation = random_rotation(3, rng=2)
    diagonal = np.diag([50.0, 5.0, 1.0])
    hessian = rotation @ diagonal @ rotation.T

    X = np.random.default_rng(3).random((40, 3)) * 2 - 1
    _, dy = quadratic(X, hessian)

    frame = IsotropicTransformer(method="ge-lhm").fit(X, dy=dy)

    diagonalised = frame.rotation_.T @ hessian @ frame.rotation_
    off_diagonal = diagonalised - np.diag(np.diag(diagonalised))
    assert np.max(np.abs(off_diagonal)) < 1e-6
    np.testing.assert_allclose(frame.eigenvalues_, [50.0, 5.0, 1.0], rtol=1e-6)


def test_ge_dlhm_keeps_only_the_diagonal(sampled):
    X, y, dy = sampled

    frame = IsotropicTransformer(method="ge-dlhm").fit(X, dy=dy)

    np.testing.assert_array_equal(frame.rotation_, np.identity(3))
    np.testing.assert_allclose(frame.eigenvalues_, np.diag(frame.curvature_))


def test_asm_is_the_gradient_covariance(sampled):
    X, y, dy = sampled

    frame = IsotropicTransformer(method="asm").fit(X, dy=dy)

    np.testing.assert_allclose(frame.curvature_, dy.T @ dy / dy.shape[0], rtol=1e-13)


def test_eigenvalues_are_descending_and_reproducible(sampled):
    X, y, dy = sampled

    for method in ("ge-lhm", "fv-lhm", "asm"):
        first = IsotropicTransformer(method=method).fit(X, y=y, dy=dy)
        second = IsotropicTransformer(method=method).fit(X, y=y, dy=dy)

        assert np.all(np.diff(first.eigenvalues_) <= 0), method
        np.testing.assert_array_equal(first.rotation_, second.rotation_)
        np.testing.assert_array_equal(first.eigenvalues_, second.eigenvalues_)


# ---------------------------------------------------------------- applying the frame


def test_transform_round_trips(sampled):
    X, y, dy = sampled
    frame = IsotropicTransformer().fit(X, dy=dy)

    np.testing.assert_allclose(frame.inverse_transform(frame.transform(X)), X, rtol=1e-10)
    np.testing.assert_allclose(
        frame.inverse_transform_gradient(frame.transform_gradient(dy)), dy, rtol=1e-10
    )


def test_transform_makes_a_known_anisotropic_quadratic_isotropic(anisotropic_quadratic):
    """The whole point: after transforming, curvature is the same in every direction."""
    X, y, dy, _ = anisotropic_quadratic
    frame = IsotropicTransformer(method="ge-lhm").fit(X, dy=dy)

    transformed = IsotropicTransformer(method="ge-lhm").fit(
        frame.transform(X), dy=frame.transform_gradient(dy)
    )

    np.testing.assert_allclose(transformed.curvature_, np.identity(2), rtol=1e-6, atol=1e-8)


def test_transformed_gradient_satisfies_the_chain_rule(sampled):
    """transform_gradient must give the gradient of the function seen in the new frame."""
    X, y, dy = sampled
    frame = IsotropicTransformer().fit(X, dy=dy)

    analytic = frame.transform_gradient(dy)

    # Differentiate x_hat -> f(inverse_transform(x_hat)) at x_hat = transform(X).
    Xt = frame.transform(X)
    step = 1e-6
    numeric = np.empty_like(Xt)
    for f in range(Xt.shape[1]):
        forward, backward = Xt.copy(), Xt.copy()
        forward[:, f] += step
        backward[:, f] -= step
        numeric[:, f] = (
            non_isotropic(frame.inverse_transform(forward))[0]
            - non_isotropic(frame.inverse_transform(backward))[0]
        ) / (2 * step)

    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-8)


def test_ideal_uses_the_supplied_rotation_and_scaling_directly(sampled):
    """The old API square-rooted this argument while documenting it as scalers."""
    X, _, _ = sampled
    rotation = random_rotation(3, rng=4)
    scaling = np.array([3.0, 2.0, 1.0])

    frame = IsotropicTransformer(method="ideal", rotation=rotation, scaling=scaling).fit(X)

    np.testing.assert_array_equal(frame.scaling_, scaling)
    np.testing.assert_array_equal(frame.rotation_, rotation)
    np.testing.assert_allclose(frame.transform(X), X @ rotation * scaling, rtol=1e-13)


# ------------------------------------------------------------------- no mutation


def test_fitting_and_transforming_leave_the_inputs_untouched(sampled):
    """The old Transform() rewrote model.X, model.C and model.dy in place."""
    X, y, dy = sampled
    X_before, y_before, dy_before = X.copy(), y.copy(), dy.copy()

    frame = IsotropicTransformer().fit(X, y=y, dy=dy)
    frame.transform(X)
    frame.transform_gradient(dy)

    np.testing.assert_array_equal(X, X_before)
    np.testing.assert_array_equal(y, y_before)
    np.testing.assert_array_equal(dy, dy_before)


def test_refitting_does_not_corrupt_the_frame(sampled):
    """The old Transform() was not idempotent: a second call destroyed the original data."""
    X, y, dy = sampled
    frame = IsotropicTransformer().fit(X, dy=dy)
    first = frame.transform(X).copy()

    frame.fit(X, dy=dy)

    np.testing.assert_array_equal(frame.transform(X), first)


def test_transforming_twice_is_not_silently_the_same_as_once(sampled):
    """Applying the frame is explicit, so double application is visible, not hidden state."""
    X, y, dy = sampled
    frame = IsotropicTransformer().fit(X, dy=dy)

    once = frame.transform(X)
    twice = frame.transform(once)

    assert not np.allclose(once, twice)


# ------------------------------------------------------------------------ guards


def test_missing_data_is_reported_per_method(sampled):
    X, y, dy = sampled

    with pytest.raises(ValueError, match="requires gradients"):
        IsotropicTransformer(method="ge-lhm").fit(X, y=y)
    with pytest.raises(ValueError, match="requires gradients"):
        IsotropicTransformer(method="asm").fit(X, y=y)
    with pytest.raises(ValueError, match="requires function values"):
        IsotropicTransformer(method="fv-lhm").fit(X, dy=dy)
    with pytest.raises(ValueError, match="requires both rotation and scaling"):
        IsotropicTransformer(method="ideal").fit(X)


def test_unknown_method_is_reported(sampled):
    X, _, dy = sampled
    with pytest.raises(ValueError, match="method must be one of"):
        IsotropicTransformer(method="pca").fit(X, dy=dy)


def test_fv_lhm_explains_how_many_samples_it_needs():
    """The quadratic fit scales badly with dimension; the error should say so."""
    X = np.random.default_rng(5).random((8, 5))
    y, _ = non_isotropic(X)

    with pytest.raises(ValueError, match="method='ge-lhm', which needs only"):
        IsotropicTransformer(method="fv-lhm").fit(X, y=y)


def test_neighbour_count_guards(sampled):
    X, y, dy = sampled

    with pytest.raises(ValueError, match="at least 1"):
        IsotropicTransformer(n_neighbors=0).fit(X, dy=dy)
    with pytest.raises(ValueError, match="needs more than"):
        IsotropicTransformer(n_neighbors=X.shape[0]).fit(X, dy=dy)


def test_ideal_rejects_a_non_orthogonal_rotation_or_bad_scaling(sampled):
    X, _, _ = sampled

    with pytest.raises(ValueError, match="orthogonal"):
        IsotropicTransformer(method="ideal", rotation=np.full((3, 3), 0.5), scaling=np.ones(3)).fit(
            X
        )
    with pytest.raises(ValueError, match="must all be positive"):
        IsotropicTransformer(method="ideal", rotation=np.identity(3), scaling=[1.0, 0.0, 1.0]).fit(
            X
        )
    with pytest.raises(ValueError, match="scaling must have shape"):
        IsotropicTransformer(method="ideal", rotation=np.identity(3), scaling=[1.0, 1.0]).fit(X)


def test_degenerate_curvature_is_reported_not_silently_nan():
    """A flat direction would give a zero scaler; sqrt of a negative would give nan."""
    X = np.random.default_rng(6).random((20, 2))
    # A function that is exactly flat in the second coordinate.
    dy = np.column_stack([2 * X[:, 0], np.zeros(X.shape[0])])

    with pytest.raises(ValueError, match="zero eigenvalue"):
        IsotropicTransformer(method="asm").fit(X, dy=dy)


def test_applying_before_fitting_raises():
    frame = IsotropicTransformer()
    with pytest.raises(NotFittedError):
        frame.transform(np.zeros((2, 2)))
    with pytest.raises(NotFittedError):
        frame.transform_gradient(np.zeros((2, 2)))


def test_feature_count_mismatch_is_reported(sampled):
    X, _, dy = sampled
    frame = IsotropicTransformer().fit(X, dy=dy)

    with pytest.raises(ValueError, match="X has 5 features"):
        frame.transform(np.zeros((4, 5)))


def test_is_a_well_behaved_sklearn_transformer(sampled):
    X, y, dy = sampled
    frame = IsotropicTransformer(method="asm", n_neighbors=4)

    assert frame.get_params()["method"] == "asm"
    assert clone(frame).get_params() == frame.get_params()
