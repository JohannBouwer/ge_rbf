# GE_RBF

Radial basis function (RBF) surrogate modelling strategies developed during my PhD research.
The work is about two things: using sampled gradient information when building RBF surrogates,
and pre-processing the sampled data into a coordinate frame where those surrogates actually
work well.

## Paper

> Johann Bouwer, Daniel N. Wilke and Schalk Kok.
> **A novel and fully automated coordinate system transformation scheme for near optimal
> surrogate construction.**
> *Computer Methods in Applied Mechanics and Engineering* **419** (2024), 116648.
> [doi.org/10.1016/j.cma.2023.116648](https://doi.org/10.1016/j.cma.2023.116648)

This package implements that paper: the transformation scheme is `IsotropicTransformer`, the
surrogates are `RBFRegressor`, and the test function of Section 5 is `problems.non_isotropic`.
[`notebooks/03_transformations.ipynb`](notebooks/03_transformations.ipynb) reproduces its central
results. Section references throughout this README and the source point at this paper.

Two earlier papers whose methods also appear here:
[Mech. Based Des. Struct. Mach. (2021)](https://doi.org/10.1080/15397734.2021.1950549) and
[Math. Comput. Appl. 28(2):57 (2023)](https://doi.org/10.3390/mca28020057).

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

- [GE\_RBF](#ge_rbf)
  - [Paper](#paper)
  - [The idea in one paragraph](#the-idea-in-one-paragraph)
  - [Contents](#contents)
  - [Installation](#installation)
  - [Quick start](#quick-start)
  - [Models](#models)
  - [Shape parameter selection](#shape-parameter-selection)
  - [Coordinate transformations](#coordinate-transformations)
  - [Test problems](#test-problems)
  - [Notebooks](#notebooks)
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
  methods compared, how the benefit holds up from 2 to 16 dimensions, and where sampled gradients
  are actually worth spending

That last section is the one to read if you read only one. Holding model flexibility fixed and
varying only what the gradients are used for, across 11 dimension/sample-size configurations:
spending them on the **coordinate frame** cuts the error by a median of **1.88×** (function-value
models) or **1.84×** (gradient-enhanced), while spending them as **extra rows in the fitting system**
buys **1.10×** — and makes the model worse in 2 of the 11. The dominant error is the mismatch
between an isotropic basis and an anisotropic response, not a shortage of information about the
function. If gradients are available, the frame and the shape-parameter search are where they pay.

The committed outputs are current — every notebook was executed top to bottom against the code as
it stands, so you can read the results without running anything.


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
