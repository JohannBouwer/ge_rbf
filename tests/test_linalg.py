"""Tests for the shared validation and eigendecomposition helpers."""

import numpy as np
import pytest

from ge_rbf._linalg import (
    check_gradients,
    check_samples,
    check_targets,
    restore_target_shape,
    sqrt_eigenvalues,
    symmetric_eigh,
)


def test_check_samples_promotes_1d_to_a_single_feature():
    out = check_samples([1.0, 2.0, 3.0])
    assert out.shape == (3, 1)


def test_check_samples_rejects_bad_input():
    with pytest.raises(ValueError, match="1- or 2-dimensional"):
        check_samples(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="at least one sample"):
        check_samples(np.zeros((0, 3)))
    with pytest.raises(ValueError, match="NaN or infinite"):
        check_samples([[1.0, np.nan]])


@pytest.mark.parametrize("shape", [(5,), (5, 1)])
def test_check_targets_accepts_both_shapes_and_records_which(shape):
    values, was_column = check_targets(np.arange(5.0).reshape(shape), 5)

    assert values.shape == (5,)
    assert was_column is (shape == (5, 1))
    assert restore_target_shape(values, was_column).shape == shape


def test_check_targets_rejects_mismatched_length_and_2d():
    with pytest.raises(ValueError, match="4 entries but there are 5"):
        check_targets(np.zeros(4), 5)
    with pytest.raises(ValueError, match="must have shape"):
        check_targets(np.zeros((5, 2)), 5)


def test_check_gradients_promotes_1d_only_when_there_is_one_feature():
    assert check_gradients(np.zeros(5), 5, 1).shape == (5, 1)
    with pytest.raises(ValueError, match="must have shape"):
        check_gradients(np.zeros(5), 5, 2)


def test_symmetric_eigh_sorts_descending_and_reconstructs():
    rng = np.random.default_rng(0)
    A = rng.random((4, 4))
    A = A + A.T

    eigenvalues, eigenvectors = symmetric_eigh(A)

    assert np.all(np.diff(eigenvalues) <= 0)
    np.testing.assert_allclose(eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T, A, atol=1e-12)
    np.testing.assert_allclose(eigenvectors.T @ eigenvectors, np.eye(4), atol=1e-12)


def test_symmetric_eigh_sign_convention_is_deterministic():
    """A sign-flipped input must give the same eigenvectors back."""
    A = np.diag([3.0, 1.0, 2.0])
    _, first = symmetric_eigh(A)
    _, second = symmetric_eigh(A.copy())

    np.testing.assert_array_equal(first, second)
    # Largest-magnitude entry of every column is positive.
    dominant = np.argmax(np.abs(first), axis=0)
    assert np.all(first[dominant, np.arange(3)] > 0)


def test_symmetric_eigh_ignores_asymmetric_round_off():
    A = np.array([[2.0, 1.0], [1.0 + 1e-15, 2.0]])
    eigenvalues, _ = symmetric_eigh(A)
    np.testing.assert_allclose(eigenvalues, [3.0, 1.0], atol=1e-12)


def test_sqrt_eigenvalues_guards_degenerate_curvature():
    np.testing.assert_allclose(sqrt_eigenvalues(np.array([4.0, 9.0])), [2.0, 3.0])

    with pytest.raises(ValueError, match="negative eigenvalues"):
        sqrt_eigenvalues(np.array([1.0, -1e-8]))
    with pytest.raises(ValueError, match="zero eigenvalue"):
        sqrt_eigenvalues(np.array([1.0, 0.0]))
