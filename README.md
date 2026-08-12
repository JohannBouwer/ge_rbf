# GE_RBF

Radial basis function (RBF) surrogate modelling strategies developed during my PhD
research ([1](https://doi.org/10.1080/15397734.2021.1950549),
[2](https://doi.org/10.3390/mca28020057),
[3](https://doi.org/10.1016/j.cma.2023.116648)). The work is about two things: using
sampled gradient information when building RBF surrogates, and pre-processing the sampled
data into a coordinate frame where those surrogates actually work well.

## The idea in one paragraph

RBF basis functions are **isotropic** — they depend only on `‖x - c‖`, so the model assumes
the response varies at the same rate in every direction. Real responses rarely do. The
mismatch shows up as model bias, and it is why adding gradient information to a surrogate
can *fail* to improve it: the extra information ends up fighting the functional form rather
than informing it. The fix is a pre-processing step that estimates a rotation and a
per-direction scaling making the sampled data roughly isotropic. Component-wise scaling
alone is not enough when the variables are coupled — a rotation is needed too, and
estimating it well needs *local* curvature estimates rather than a global one.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Models](#models)
- [Shape parameter selection](#shape-parameter-selection)
- [Coordinate transformations](#coordinate-transformations)
- [Test problems](#test-problems)
- [Notebooks](#notebooks)
- [Implementation notes vs. the paper](#implementation-notes-vs-the-paper)
- [Development](#development)
- [Citation](#citation)

## Installation

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/JohannBouwer/GE_RBF && cd GE_RBF && uv sync
```

## Quick start

```python
import numpy as np
from ge_rbf import IsotropicTransformer, RBFRegressor, gradient_search
from ge_rbf.problems import non_isotropic

X = np.random.default_rng(0).random((30, 2))
y, dy = non_isotropic(X)

# Estimate a coordinate frame in which the sampled response is closer to isotropic.
frame = IsotropicTransformer(method="ge-lhm").fit(X, dy=dy)

# Fit in that frame, letting the search pick the shape parameter.
result = gradient_search(RBFRegressor(), frame.transform(X), y, frame.transform_gradient(dy))
model = result.best_estimator

# Predict, then map the gradients back to the original coordinates.
probe = frame.transform(np.random.default_rng(1).random((10, 2)))
y_hat, dy_hat = model.predict(probe, return_gradient=True)
dy_hat = frame.inverse_transform_gradient(dy_hat)
```

The API follows scikit-learn conventions: estimators are constructed with
hyperparameters, `fit` returns `self`, fitted state carries a trailing underscore, and
nothing mutates its inputs.

## Models

`RBFRegressor` covers all three model types. Which one you get follows from the data you
pass to `fit`:

| Call | Model |
| --- | --- |
| `fit(X, y)` | **FV** — fitted to function values only |
| `fit(X, y, dy=dy)` | **GE** — gradient-enhanced, fitted to both |
| `fit(X, dy=dy, anchor_X=..., anchor_y=...)` | **GO** — gradient-only, anchored |

Gradients alone determine a surrogate only up to an additive constant, which is why the
gradient-only model needs an anchor: one location and the function value there. The anchor
need not be a sampled point.

Placing one centre per sample (the default) makes the function-value system square and
interpolating. Passing fewer `centers` gives a regression fit, which is cheaper and less
prone to fitting noise.

## Shape parameter selection

The shape parameter `epsilon` controls how wide the basis functions are, and it is the one
hyperparameter that matters. All three searches sweep a range of candidates, reject those
whose system is too ill-conditioned to trust, and return a `SearchResult` carrying the
winner and a model fitted at it. They differ in where the error signal comes from:

| Function | Error signal |
| --- | --- |
| `kfold_search` | Held-out folds of the sampled data |
| `validation_search` | A separate validation set |
| `gradient_search` | The sampled gradients, with nothing held out |

`gradient_search` is the interesting one: with `use_gradients_in_fit=False` it fits
function-value models and scores them against the sampled gradients, which are then a
genuinely independent signal needing no extra function evaluations.

`plot_search(result)` draws the error curve.

## Coordinate transformations

`IsotropicTransformer` estimates the frame and applies it. It never modifies the estimator
or the data it is given.

| `method` | Curvature estimate | Points per estimate |
| --- | --- | --- |
| `"ge-lhm"` | Local Hessians from gradients (SR1 updates) | `n + 1` |
| `"ge-dlhm"` | As above, diagonal only — scaling, no rotation | `n + 1` |
| `"fv-lhm"` | Local quadratic fits to function values | `n(n+1)/2 + n + 1` |
| `"asm"` | Active subspace: averaged gradient outer product | all samples |
| `"ideal"` | A rotation and scaling you supply | — |

`"ge-lhm"` is the recommended method when gradients are available: it scales linearly with
dimension where the function-value alternative scales quadratically.

Rotating and scaling the coordinates changes what a gradient *means*, so sampled gradients
must be converted with `transform_gradient` before they are used to fit in the new frame,
and predicted gradients converted back with `inverse_transform_gradient`.

## Test problems

`ge_rbf.problems` provides `non_isotropic`, `rosenbrock`, `rastrigin`, `ackley`, `sphere`
and `beale`, each returning function values of shape `(n_samples,)` and gradients of shape
`(n_samples, n_features)`.

`non_isotropic` is the paper's test problem: decomposable by construction, so the ideal
frame is known exactly, with a deliberately different length scale per direction. Passing a
`rotation` (see `random_rotation`) couples the variables, so a transformation scheme has to
recover both a rotation and a scaling.

## Notebooks

```bash
uv run jupyter lab
```

- [`notebooks/01_models.ipynb`](notebooks/01_models.ipynb) — the three model types
- [`notebooks/02_shape_parameter.ipynb`](notebooks/02_shape_parameter.ipynb) — the three searches
- [`notebooks/03_transformations.ipynb`](notebooks/03_transformations.ipynb) — the transformation
  methods compared, plus a study of how the benefit holds up from 2 to 16 dimensions

A note on reading that last one. Comparing accuracy across dimensions needs a rule for how many
samples each dimension gets, and **no affordable rule is fair**. RBF accuracy tracks the fill
distance, which scales as `n^(-1/d)`, so matching the sample density of a 20-point 2-D design would
take 2.6 × 10¹⁰ points in 16-D. Going from `n = 5d` to `n = d(d+1)/2` at 16 dimensions raises the
count by 70% and tightens the spacing by 3% — every polynomial budget is asymptotically the same
budget. Absolute errors are therefore not comparable across dimensions; the ratio between frames at
matched sample count is. The notebook budgets by `n = c·d(d+1)/2` — one multiple of the number of
free parameters in a symmetric Hessian — because that holds the *frame-estimation* problem equally
determined across dimensions, which is the thing the study is actually about.

Committed outputs are current: the notebooks are executed as part of the release checks.

## Implementation notes vs. the paper

Three points where the code deliberately departs from the published description. All three
were in the original implementation and are kept so that results reproduce; they are
recorded here because the paper alone would not predict them.

1. **Combining local Hessians.** Eq. 20 specifies the *mean* of the reconstructed local
   Hessians. The code uses an element-wise **median**, which is far less sensitive to a
   single badly conditioned local estimate. This is not cosmetic — on a 4-D `non_isotropic`
   problem the two give normalised eigenvalue spectra differing by up to 0.8.
2. **SR1 initialisation.** Procedure 2 line 2 specifies `H₀ = I`. The code initialises each
   local estimate from the outer product of the gradient at that point, `∇f ∇fᵀ`.
3. **SR1 early exit.** The code stops a sweep if any component of the SR1 numerator falls
   below `1e-6`, guarding against a vanishing denominator. This guard is not in the paper.
   It never triggered across 9 000 updates on the problems studied there.

One point is **unresolved** and worth checking against your copy of the paper. Eq. 32 reads
`Aᵢ = -2·exp(-(2i-N)²/N) + 3`, whereas `non_isotropic` implements
`-2·exp(-(2i-N)²/(2N)) + 3` — a factor of two in the Gaussian width. Both satisfy the
stated `[1, 3]` bound, and the code's form is what produced the published results, so the
implementation is unchanged pending confirmation. The frequency (Eq. 33) matches: the
code's sigmoid gives exactly the stated `[0.5π, 2π]` range.

## Development

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

`tests/test_legacy_regression.py` checks the package against numerical output captured from
the pre-refactor implementation, so the research results are pinned. Two places
intentionally differ and are asserted as bounded differences: gradient-only fits now go
through `lstsq` rather than the normal equations `KᵀK w = KᵀY`, which squared the condition
number of an already delicate system; and eigendecompositions are now canonically ordered
and signed, so transformed *coordinates* can be permuted or sign-flipped. Neither is
visible in a prediction — the kernel depends only on Euclidean distance, which neither
operation changes.

## Citation

```bibtex
@article{bouwer2024coordinate,
  title   = {A novel and fully automated coordinate system transformation scheme
             for near optimal surrogate construction},
  author  = {Bouwer, Johann and Wilke, Daniel N. and Kok, Schalk},
  journal = {Computer Methods in Applied Mechanics and Engineering},
  volume  = {419},
  pages   = {116648},
  year    = {2024},
  doi     = {10.1016/j.cma.2023.116648}
}
```
