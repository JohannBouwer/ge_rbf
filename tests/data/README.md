# Regression fixtures

## `legacy_baseline.npz`

Numerical output captured from the **pre-refactor** implementation
(`src/ge_rbf/rbf_models.py` and `src/ge_rbf/preprocessing.py` as they stood at commit
`4259267`, the last commit on `main` before the rewrite).

It exists so `tests/test_legacy_regression.py` can prove that the rewrite reproduces the
published behaviour bit-for-bit on every code path that actually worked. Paths that were
broken in the old code (`FV_LHM`, rotated `NonIso`, function-value-only `Transform`,
non-square `FV_fit`) are deliberately absent — there was nothing correct to preserve.

The arrays cover, for 2-D and 4-D `NonIso` datasets on fixed seeds:

- `*_fv_*`, `*_ge_*`, `*_go_*` — fitted coefficients, kernel condition number, and
  predicted function values / gradients at held-out points.
- `*_gelhm_*`, `*_asm_*` — estimated Hessian / covariance matrix and its eigendecomposition.
- `*_tr_*` — the transformed samples, centres, gradients and scalers for each
  transformation method, plus a model fitted in the transformed frame.
- `*_gradval_*`, `*_valset_*` — shape parameters selected by the deterministic searches.
- `prob_*` — function and gradient values from the test problems. Note that
  `prob_ackley_g` is **not** stored: the old Ackley gradient was analytically wrong, so
  reproducing it would be a bug, not a regression guard.

The generator script is not kept in the repository because it imports modules that the
refactor deleted. To regenerate, check out `main` and run the script recorded in the
audit plan.
