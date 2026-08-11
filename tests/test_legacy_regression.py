"""The rewrite must reproduce the published implementation's numbers.

``tests/data/legacy_baseline.npz`` was captured from the pre-refactor code (see the README
in that directory). Every path that worked before must still give the same answer, so that
fixing the bugs around the edges did not quietly change the research results.

Two places intentionally differ, and are asserted as *bounded* differences rather than
exact matches:

- **Gradient-only fits.** The old code solved the normal equations ``K'K w = K'Y``, which
  squares the condition number of an already delicate system. This now goes through
  ``lstsq``. The old answer is the less accurate one.
- **Coordinate frames.** Eigenvalues are now sorted and eigenvector signs fixed, so the
  transformed coordinates can come out permuted or sign-flipped relative to before. That
  is invisible to the model: the kernel depends only on Euclidean distance, which neither
  operation changes. The test therefore compares predictions in the original frame, which
  must match to machine precision.
"""

from pathlib import Path

import numpy as np
import pytest

from ge_rbf import IsotropicTransformer, RBFRegressor, gradient_search, validation_search
from ge_rbf.problems import beale, non_isotropic, rastrigin, rosenbrock, sphere

BASELINE = np.load(Path(__file__).parent / "data" / "legacy_baseline.npz")

# (n_features, n_samples) — matches how the baseline was generated.
CASES = [(2, 15), (4, 40)]


def dataset(n_features, n_samples):
    """Reproduce a baseline dataset, including the draw order of the original script."""
    rng = np.random.default_rng(1234 + n_features)
    X = rng.random((n_samples, n_features))
    y, dy = non_isotropic(X)
    X_test = rng.random((23, n_features))
    X_valid = rng.random((30, n_features))
    y_valid, _ = non_isotropic(X_valid)
    return X, y, dy, X_test, X_valid, y_valid


@pytest.mark.parametrize(("n_features", "n_samples"), CASES)
def test_function_value_model_is_unchanged(n_features, n_samples):
    X, y, _, X_test, _, _ = dataset(n_features, n_samples)
    tag = f"d{n_features}"

    model = RBFRegressor(epsilon=0.8).fit(X, y)
    predicted, predicted_gradient = model.predict(X_test, return_gradient=True)

    np.testing.assert_allclose(model.coef_, BASELINE[f"{tag}_fv_coef"].ravel(), rtol=1e-10)
    np.testing.assert_allclose(predicted, BASELINE[f"{tag}_fv_pred_y"].ravel(), rtol=1e-10)
    np.testing.assert_allclose(predicted_gradient, BASELINE[f"{tag}_fv_pred_dy"], rtol=1e-10)


@pytest.mark.parametrize(("n_features", "n_samples"), CASES)
def test_gradient_enhanced_model_is_unchanged(n_features, n_samples):
    X, y, dy, X_test, _, _ = dataset(n_features, n_samples)
    tag = f"d{n_features}"

    model = RBFRegressor(epsilon=0.8).fit(X, y, dy=dy)
    predicted, predicted_gradient = model.predict(X_test, return_gradient=True)

    np.testing.assert_allclose(model.coef_, BASELINE[f"{tag}_ge_coef"].ravel(), rtol=1e-10)
    np.testing.assert_allclose(predicted, BASELINE[f"{tag}_ge_pred_y"].ravel(), rtol=1e-10)
    np.testing.assert_allclose(predicted_gradient, BASELINE[f"{tag}_ge_pred_dy"], rtol=1e-10)


@pytest.mark.parametrize(("n_features", "n_samples"), CASES)
def test_gradient_only_model_agrees_to_the_accuracy_of_the_old_solver(n_features, n_samples):
    """Bounded difference: normal equations replaced by least squares. See module docstring."""
    X, y, dy, X_test, _, _ = dataset(n_features, n_samples)
    tag = f"d{n_features}"

    model = RBFRegressor(epsilon=0.8).fit(X, dy=dy, anchor_X=X[:1], anchor_y=y[:1])
    predicted, predicted_gradient = model.predict(X_test, return_gradient=True)

    np.testing.assert_allclose(predicted, BASELINE[f"{tag}_go_pred_y"].ravel(), rtol=1e-5)
    np.testing.assert_allclose(predicted_gradient, BASELINE[f"{tag}_go_pred_dy"], rtol=1e-5)


@pytest.mark.parametrize(("n_features", "n_samples"), CASES)
def test_curvature_estimates_are_unchanged(n_features, n_samples):
    X, y, dy, _, _, _ = dataset(n_features, n_samples)
    tag = f"d{n_features}"

    ge_lhm = IsotropicTransformer(method="ge-lhm").fit(X, dy=dy)
    np.testing.assert_allclose(ge_lhm.curvature_, BASELINE[f"{tag}_gelhm_H"], rtol=1e-12)
    np.testing.assert_allclose(
        ge_lhm.eigenvalues_, np.sort(BASELINE[f"{tag}_gelhm_eig"])[::-1], rtol=1e-12
    )

    asm = IsotropicTransformer(method="asm").fit(X, dy=dy)
    np.testing.assert_allclose(asm.curvature_, BASELINE[f"{tag}_asm_C"], rtol=1e-12)
    np.testing.assert_allclose(
        asm.eigenvalues_, np.sort(BASELINE[f"{tag}_asm_eig"])[::-1], rtol=1e-12
    )


@pytest.mark.parametrize(("n_features", "n_samples"), CASES)
@pytest.mark.parametrize(
    ("method", "key"), [("ge-lhm", "GELHM"), ("ge-dlhm", "GEDLHM"), ("asm", "ASM")]
)
def test_predictions_from_a_transformed_frame_are_unchanged(n_features, n_samples, method, key):
    X, y, dy, X_test, _, _ = dataset(n_features, n_samples)
    tag = f"d{n_features}"

    frame = IsotropicTransformer(method=method).fit(X, y=y, dy=dy)
    model = RBFRegressor(epsilon=0.8).fit(frame.transform(X), y, dy=frame.transform_gradient(dy))

    transformed_test = frame.transform(X_test)
    predicted = model.predict(transformed_test)
    predicted_gradient = frame.inverse_transform_gradient(model.predict_gradient(transformed_test))

    np.testing.assert_allclose(predicted, BASELINE[f"{tag}_tr_{key}_pred_y"].ravel(), rtol=1e-10)
    np.testing.assert_allclose(predicted_gradient, BASELINE[f"{tag}_tr_{key}_pred_dy"], rtol=1e-10)


@pytest.mark.parametrize(("n_features", "n_samples"), CASES)
def test_shape_parameter_searches_select_the_same_value(n_features, n_samples):
    """Compared with the conditioning filter disabled: the old filter never bound."""
    X, y, dy, _, X_valid, y_valid = dataset(n_features, n_samples)
    tag = f"d{n_features}"
    epsilons = np.logspace(-2, 1, 25)

    for use_gradients, label in ((True, "ge"), (False, "fv")):
        fit_gradients = dy if use_gradients else None

        found = gradient_search(
            RBFRegressor(),
            X,
            y,
            dy,
            use_gradients_in_fit=use_gradients,
            epsilons=epsilons,
            max_condition=np.inf,
        )
        assert found.best_epsilon == pytest.approx(float(BASELINE[f"{tag}_gradval_{label}"]))

        found = validation_search(
            RBFRegressor(),
            X,
            y,
            X_valid,
            y_valid,
            dy=fit_gradients,
            epsilons=epsilons,
            max_condition=np.inf,
        )
        assert found.best_epsilon == pytest.approx(float(BASELINE[f"{tag}_valset_{label}"]))


@pytest.mark.parametrize(
    ("problem", "legacy_name"),
    [
        (non_isotropic, "NonIso"),
        (rosenbrock, "Rosenbach"),
        (rastrigin, "rastrigin"),
        (sphere, "sphere"),
        (beale, "beale"),
    ],
)
def test_test_problem_values_are_unchanged(problem, legacy_name):
    """Ackley is absent: its old gradient was wrong, so there is nothing to preserve."""
    X = np.random.default_rng(7).random((11, 2)) * 4 - 2

    f, grad = problem(X)

    np.testing.assert_allclose(f, BASELINE[f"prob_{legacy_name}_f"].ravel(), rtol=1e-13)
    np.testing.assert_allclose(grad, BASELINE[f"prob_{legacy_name}_g"], rtol=1e-13)
