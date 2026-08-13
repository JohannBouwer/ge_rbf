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
  - [Trajectory data](#trajectory-data)
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

Errors and warnings follow one rule: `ValueError` for anything the caller can fix, and
`RuntimeWarning` when the package has silently done something less good than what was
asked for. So `warnings.filterwarnings("error", category=RuntimeWarning)` turns every
quiet degradation into a failure.

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

`centers` also accepts an **integer**, meaning "draw this many space-filling centres over
the bounding box of the samples", at fit time. The box comes from the samples rather than
from any declared bounds, because a design of experiments does not fill its box in three or
more dimensions and a centre in a region nothing was sampled from only adds an
ill-conditioned column. The draw happens in whatever coordinates `fit` was given — after
any frame — since that is where the kernel measures distance. `random_state` makes it
reproducible, and `basis_search` chooses the count.

This matters when the samples are clustered. One centre per sample hands the basis exactly
the anisotropy the samples have; re-sampling decouples *where the response is sampled* from
*where it is represented*.

## Shape parameter selection

The shape parameter `epsilon` controls how wide the basis functions are, and for a fixed
basis it is the one hyperparameter that matters. Every search sweeps a range of candidates,
rejects those whose system is too ill-conditioned to trust, and returns a result carrying
the winner and a model fitted at it. They differ in where the error signal comes from:

| Function | Error signal | Searches |
| --- | --- | --- |
| `kfold_search` | Held-out folds of the sampled data | `epsilon` |
| `validation_search` | A separate validation set | `epsilon` |
| `gradient_search` | The sampled gradients, with nothing held out | `epsilon` |
| `basis_search` | Held-out folds, or the sampled gradients | `epsilon` **and** the centre count |

`gradient_search` is the interesting one: with `use_gradients_in_fit=False` it fits
function-value models and scores them against the sampled gradients, which are then a
genuinely independent signal needing no extra function evaluations.

`basis_search` exists because once centres are re-sampled the count matters as much as the
width, and the two interact — the best `epsilon` moves by an order of magnitude across
centre counts, so searching either alone finds the best value given a bad choice of the
other.

It is the one search here that holds data out by default (`score="kfold"`), because a
training-residual score cannot choose a centre count. A gradient-enhanced model has the
sampled gradients as rows of its own fitting system, so it reproduces them better the more
centres it is given, and the score falls with the count monotonically — it can only ever
pick the largest option. `score="gradient"` fits function values only, which fixes that,
and is `k + 1` times cheaper; it measured meaningfully worse on clustered data, so it is
the option rather than the default.

`kfold_search` and `basis_search` both take `groups`, holding out whole groups instead of
individual rows. Use it whenever samples come in tight clusters — otherwise a held-out
row's nearest neighbours are still in the training set, the fold error is close to an
interpolation error, and the search will happily choose a basis far larger than the data
supports.

`scaled_epsilons(X)` expresses the candidates relative to the sample spacing. The Gaussian
is `exp(-epsilon * ‖x - c‖²)`, so `epsilon` multiplies the **squared** distance and the
scale-invariant form divides by `h²`, not `h`, where `h` is the median distance between
samples. Reach for it whenever the coordinates are not of order one, which after a frame
they are not — the failure is quiet rather than loud, every basis function underflowing
until the model predicts a flat zero.

`plot_search(result)` draws the error curve; `plot_basis_search(result)` draws the joint
search as a heatmap, and `values="condition"` draws the conditioning over the same grid.

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
| `"identity"` | The frame that does nothing | — |

`"ge-lhm"` is the recommended method when gradients are available: it scales linearly with
dimension where the function-value alternative scales quadratically.

Rotating and scaling the coordinates changes what a gradient *means*, so sampled gradients
must be converted with `transform_gradient` before they are used to fit in the new frame,
and predicted gradients converted back with `inverse_transform_gradient`.

Sometimes there is no frame to find. The scalers are the square roots of the curvature
estimate's eigenvalues, so an estimate that is not positive definite does not describe a
frame and `sqrt_eigenvalues` refuses it rather than guessing. Local Hessians built from
gradients are positive definite only where the response is locally convex, which a
trajectory through a limit point need not be — and whether it happens depends on the sample
set, so it cannot be settled once before fitting. `fallback=True` retreats
`ge-lhm` → `ge-dlhm` → `asm` → `identity`, warns at each step, and records what actually
worked on `method_`. Only a degenerate curvature estimate triggers the retreat: a missing
`dy` or a wrong shape still raises, so a typo cannot be mistaken for a response that has no
frame. `asm` sits last because it has measured *worse than no frame at all* on one real
dataset — 6881× anisotropy against `ge-lhm`'s 21× — which makes it a last resort rather
than a good default. A method that can almost always return *a* frame is the one you want
at the end of a retreat and the one you should not reach for first.

## Trajectory data

A continuation or transient solver does not return scattered points. It returns a handful
of **trajectories** — one per design, each a chain of closely spaced states parametrised by
a time-like coordinate: arc length along a load path, time in a transient, a continuation
parameter. Every stored point carries a gradient in all inputs, that coordinate included,
so one solve contributes as many rows as it took steps.

Stacked into one block, that is a very particular clustering. Normalise the inputs and the
samples lie on a few long thin curves: neighbouring designs might be `1/3` apart while
consecutive points on one trajectory are `1/17` apart. Every point's nearest neighbours are
then the points before and after it on its own trajectory, and nothing else — so `ge-lhm`,
which builds each local Hessian from `n + 1` neighbours, estimates the curvature across
designs from no spread whatever. Often there is no usable frame to be had at all.

`TrajectoryScaler` stretches the trajectory axis until one step along it is `stretch` times
the distance between neighbouring designs, which puts the adjacent designs back within
reach of a neighbour search. The factor is a ratio of a design distance to a trajectory
distance, so it is unit-free: rescaling that coordinate cancels exactly, and only the
design columns need to be on comparable scales with each other beforehand.

```python
from ge_rbf import IsotropicTransformer, RBFRegressor, TrajectoryScaler, basis_search
from ge_rbf.problems import load_path, load_path_samples

Z, groups = load_path_samples()      # (n, n_variables + 1), trajectory coordinate last
y, dy = load_path(Z)

# 1. Make the trajectory axis commensurate with the design spacing.
scaler = TrajectoryScaler().fit(Z, groups=groups)
Zs, dys = scaler.transform(Z), scaler.transform_gradient(dy)

# 2. Estimate the frame there, retreating if this sample set has no frame to find.
frame = IsotropicTransformer(fallback=True).fit(Zs, dy=dys)
Zt, dyt = frame.transform(Zs), frame.transform_gradient(dys)

# 3. Choose how many basis functions and how wide, holding out whole trajectories.
result = basis_search(RBFRegressor(), Zt, y, dyt, groups=groups)
model = result.best_estimator

# 4. Predict, and bring the gradients back through both maps, in order.
y_hat, dy_hat = model.predict(Zt, return_gradient=True)
dy_hat = scaler.inverse_transform_gradient(frame.inverse_transform_gradient(dy_hat))
```

Three things to get right:

- **Grouping.** `groups=None` infers trajectories by comparing design rows exactly, which
  works if the block was built with `np.repeat` and fails if each row carries its own
  round-off. Pass `groups` explicitly for anything that came back from a solver.
- **Units.** Put the design variables on comparable scales first — a Euclidean distance
  across mixed units is dominated by whichever has the largest range. The trajectory column
  needs no such treatment.
- **Gradient weighting.** Use `RBFRegressor(gradient_weight="auto")` when the response and
  its derivatives differ by orders of magnitude, as a load factor of order 100 sitting
  beside a derivative of order 0.1 does. The package default weights them equally.

## Test problems

`ge_rbf.problems` provides `non_isotropic`, `rosenbrock`, `rastrigin`, `ackley`, `sphere`
and `beale`, each returning function values of shape `(n_samples,)` and gradients of shape
`(n_samples, n_features)`.

`non_isotropic` is the paper's test problem: decomposable by construction, so the ideal
frame is known exactly, with a deliberately different length scale per direction. Passing a
`rotation` (see `random_rotation`) couples the variables, so a transformation scheme has to
recover both a rotation and a scaling.

`load_path_samples` and `load_path` are the trajectory pair, kept separate so the sampling
geometry and the response can be varied independently. The geometry reproduces the
clustering measured on real load-path data; the response has a limit point sitting at a
different arc length for every design, so the curvature along a trajectory changes sign,
and its frequency varies with the first design variable, so there is genuine curvature
coupling between a design axis and the trajectory axis for a scheme to find. Without that
last property, scaling the trajectory axis would be cosmetic.

## Notebooks

```bash
uv run jupyter lab
```

- [`notebooks/01_models.ipynb`](notebooks/01_models.ipynb) — the three model types
- [`notebooks/02_shape_parameter.ipynb`](notebooks/02_shape_parameter.ipynb) — the three searches
- [`notebooks/03_transformations.ipynb`](notebooks/03_transformations.ipynb) — the transformation
  methods compared, how the benefit holds up from 2 to 16 dimensions, and where sampled gradients
  are actually worth spending
- [`notebooks/04_trajectories.ipynb`](notebooks/04_trajectories.ipynb) — fitting a surrogate to a
  family of solved trajectories, end to end: scaling the trajectory axis, estimating the frame,
  searching the basis, and mapping predicted gradients back through both transforms

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
