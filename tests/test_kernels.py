"""Tests for the basis functions.

The row ordering of ``Kernel.gradient`` is the load-bearing convention in this package:
models stack gradient observations into the right-hand side by flattening an
``(n_samples, n_features)`` array in Fortran order, and those rows must line up with the
kernel derivative rows. ``test_gradient_row_ordering_is_feature_major`` is the test that
fails loudly if anyone changes one side without the other.
"""

import numpy as np
import pytest

from ge_rbf.kernels import GaussianKernel, get_kernel


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.random((7, 3))
    C = rng.random((5, 3))
    return X, C


def test_values_match_the_analytic_definition(data):
    X, C = data
    kernel = GaussianKernel(epsilon=0.7)

    values = kernel(X, C)

    expected = np.empty((X.shape[0], C.shape[0]))
    for i, x in enumerate(X):
        for j, c in enumerate(C):
            expected[i, j] = np.exp(-0.7 * np.sum((x - c) ** 2))

    assert values.shape == (7, 5)
    np.testing.assert_allclose(values, expected, rtol=1e-14)


def test_value_is_one_at_zero_distance():
    X = np.array([[1.0, 2.0]])
    np.testing.assert_allclose(GaussianKernel(epsilon=3.0)(X, X), [[1.0]], rtol=1e-15)


def test_gradient_row_ordering_is_feature_major(data):
    """Row ``f * n_samples + i`` must hold d phi(x_i, c_j) / d x_f."""
    X, C = data
    kernel = GaussianKernel(epsilon=0.7)
    n_samples, n_features = X.shape

    gradient = kernel.gradient(X, C)
    assert gradient.shape == (n_samples * n_features, C.shape[0])

    for f in range(n_features):
        for i in range(n_samples):
            expected = -2 * 0.7 * (X[i, f] - C[:, f]) * kernel(X[i : i + 1], C)[0]
            np.testing.assert_allclose(gradient[f * n_samples + i], expected, rtol=1e-13)


def test_gradient_rows_round_trip_through_fortran_reshape(data):
    """The inverse of the stacking convention: reshaping back must recover per-sample rows."""
    X, C = data
    kernel = GaussianKernel(epsilon=1.3)
    n_samples, n_features = X.shape

    weights = np.random.default_rng(1).random((C.shape[0], 1))
    stacked = kernel.gradient(X, C) @ weights
    per_sample = stacked.reshape(n_samples, n_features, order="F")

    # Compare against evaluating one sample at a time.
    for i in range(n_samples):
        row = kernel.gradient(X[i : i + 1], C) @ weights
        np.testing.assert_allclose(per_sample[i], row.ravel(), rtol=1e-13)


def test_gradient_matches_finite_differences(data):
    X, C = data
    kernel = GaussianKernel(epsilon=0.9)
    n_samples, n_features = X.shape
    step = 1e-6

    analytic = kernel.gradient(X, C).reshape(n_features, n_samples, -1)

    for f in range(n_features):
        forward, backward = X.copy(), X.copy()
        forward[:, f] += step
        backward[:, f] -= step
        numeric = (kernel(forward, C) - kernel(backward, C)) / (2 * step)
        np.testing.assert_allclose(analytic[f], numeric, rtol=1e-6, atol=1e-9)


def test_rejects_non_positive_epsilon():
    for bad in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="positive finite"):
            GaussianKernel(epsilon=bad)


def test_get_kernel_resolves_names_and_instances():
    assert isinstance(get_kernel("gaussian", 2.0), GaussianKernel)
    assert get_kernel("gaussian", 2.0).epsilon == 2.0

    # An instance is rebuilt at the requested epsilon, never reused with a stale one.
    rebuilt = get_kernel(GaussianKernel(epsilon=0.1), 5.0)
    assert rebuilt.epsilon == 5.0


def test_get_kernel_rejects_unknown_names_and_types():
    with pytest.raises(ValueError, match="Unknown kernel"):
        get_kernel("multiquadric", 1.0)
    with pytest.raises(TypeError):
        get_kernel(42, 1.0)
