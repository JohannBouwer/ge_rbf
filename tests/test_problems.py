"""Tests for the analytic test problems.

Every gradient is checked against central finite differences of the function returned by
the *same call*. That is the check the earlier implementation of ``ackley`` failed, and it
is also what catches a function and its gradient being defined in different coordinate
frames (which is what happened to the rotated ``non_isotropic`` problem).
"""

import numpy as np
import pytest

from ge_rbf.problems import (
    ackley,
    beale,
    non_isotropic,
    random_rotation,
    rastrigin,
    rosenbrock,
    sphere,
)

# (function, number of features) — Beale is 2-D only, Rosenbrock needs at least 2.
PROBLEMS = [
    (non_isotropic, 2),
    (non_isotropic, 5),
    (rosenbrock, 2),
    (rosenbrock, 4),
    (rastrigin, 3),
    (ackley, 3),
    (sphere, 4),
    (beale, 2),
]


def finite_difference_gradient(func, X, step=1e-6):
    """Central-difference Jacobian of a problem function, shape (n_samples, n_features)."""
    numeric = np.empty_like(X)
    for f in range(X.shape[1]):
        forward, backward = X.copy(), X.copy()
        forward[:, f] += step
        backward[:, f] -= step
        numeric[:, f] = (func(forward)[0] - func(backward)[0]) / (2 * step)
    return numeric


@pytest.mark.parametrize(("problem", "n_features"), PROBLEMS)
def test_shapes_are_uniform(problem, n_features):
    X = np.random.default_rng(0).random((6, n_features))
    f, grad = problem(X)

    assert f.shape == (6,), f"{problem.__name__} must return (n_samples,) function values"
    assert grad.shape == (6, n_features)


@pytest.mark.parametrize(("problem", "n_features"), PROBLEMS)
def test_gradient_matches_finite_differences(problem, n_features):
    # Kept away from the origin: Ackley's radial term is not differentiable there.
    X = np.random.default_rng(1).random((6, n_features)) * 2 + 0.5

    _, analytic = problem(X)
    numeric = finite_difference_gradient(problem, X)

    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-7)


def test_ackley_gradient_is_not_the_old_incorrect_expression():
    """Regression guard for the missing 1/(2*sqrt(s1/d)) factor in the radial term."""
    X = np.random.default_rng(2).random((5, 3)) + 0.5
    a, b, d = 20.0, 0.2, 3

    _, analytic = ackley(X)

    sum_squares = np.sum(X**2, axis=1)
    old = (a * b / d) * np.exp(-b * np.sqrt(sum_squares / d))[:, None] * (2 * X) + (
        2 * np.pi / d
    ) * np.exp(np.sum(np.cos(2 * np.pi * X), axis=1) / d)[:, None] * np.sin(2 * np.pi * X)

    assert not np.allclose(analytic, old), "ackley gradient still matches the incorrect version"
    np.testing.assert_allclose(analytic, finite_difference_gradient(ackley, X), rtol=1e-6)


def test_ackley_gradient_is_finite_at_the_origin():
    _, grad = ackley(np.zeros((1, 3)))
    assert np.all(np.isfinite(grad))
    np.testing.assert_allclose(grad, 0.0, atol=1e-12)


def test_non_isotropic_is_decomposable_without_a_rotation():
    """No rotation means a diagonal Hessian, which is what makes the ideal frame known."""
    X = np.random.default_rng(3).random((1, 4))
    step = 1e-5
    n = X.shape[1]

    hessian = np.empty((n, n))
    for j in range(n):
        forward, backward = X.copy(), X.copy()
        forward[:, j] += step
        backward[:, j] -= step
        hessian[:, j] = (non_isotropic(forward)[1] - non_isotropic(backward)[1])[0] / (2 * step)

    off_diagonal = hessian - np.diag(np.diag(hessian))
    assert np.max(np.abs(off_diagonal)) < 1e-6


def test_non_isotropic_rotation_is_consistent_between_function_and_gradient():
    """The bug this guards: the function was evaluated unrotated while gradients were rotated."""
    n = 3
    rotation = random_rotation(n, rng=0)
    X = np.random.default_rng(4).random((5, n))

    _, analytic = non_isotropic(X, rotation)
    numeric = finite_difference_gradient(lambda Z: non_isotropic(Z, rotation), X)

    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-8)


def test_non_isotropic_rotation_reproduces_the_unrotated_function():
    """f(x) in the rotated frame equals f(z) in the decomposable frame at z = x @ R."""
    n = 4
    rotation = random_rotation(n, rng=1)
    X = np.random.default_rng(5).random((6, n))

    rotated, _ = non_isotropic(X, rotation)
    plain, _ = non_isotropic(X @ rotation)

    np.testing.assert_allclose(rotated, plain, rtol=1e-13)


def test_non_isotropic_rejects_bad_rotations():
    X = np.random.default_rng(6).random((4, 3))

    with pytest.raises(ValueError, match="must have shape"):
        non_isotropic(X, np.eye(2))
    with pytest.raises(ValueError, match="orthogonal"):
        non_isotropic(X, np.full((3, 3), 0.5))


def test_random_rotation_is_orthogonal_and_seeded():
    R = random_rotation(5, rng=7)

    np.testing.assert_allclose(R.T @ R, np.eye(5), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-12)
    np.testing.assert_array_equal(R, random_rotation(5, rng=7))


def test_known_function_values():
    np.testing.assert_allclose(sphere(np.array([[3.0, 4.0]]))[0], [25.0])
    np.testing.assert_allclose(rosenbrock(np.array([[1.0, 1.0]]))[0], [0.0], atol=1e-12)
    np.testing.assert_allclose(rastrigin(np.zeros((1, 3)))[0], [0.0], atol=1e-12)
    np.testing.assert_allclose(ackley(np.zeros((1, 3)))[0], [0.0], atol=1e-12)
    np.testing.assert_allclose(beale(np.array([[3.0, 0.5]]))[0], [0.0], atol=1e-12)


def test_dimension_guards():
    with pytest.raises(ValueError, match="at least 2 variables"):
        rosenbrock(np.zeros((3, 1)))
    with pytest.raises(ValueError, match="only defined for 2 variables"):
        beale(np.zeros((3, 3)))
