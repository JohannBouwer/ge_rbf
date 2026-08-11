"""Gradient-enhanced radial basis function surrogates with isotropic coordinate transforms.

Implements the surrogate modelling strategies developed in:

- Bouwer, Wilke & Kok (2024), *A novel and fully automated coordinate system transformation
  scheme for near optimal surrogate construction*, Comput. Methods Appl. Mech. Engrg.
  https://doi.org/10.1016/j.cma.2023.116648
- Bouwer, Wilke & Kok (2023), *Gradient-only surrogate to resolve learning rates for robust
  and consistent training of deep neural networks*, Math. Comput. Appl.
  https://doi.org/10.3390/mca28020057
- Bouwer, Wilke & Kok (2021), Mech. Based Des. Struct. Mach.
  https://doi.org/10.1080/15397734.2021.1950549

Typical use: choose a shape parameter, fit a surrogate, and optionally do both in a
coordinate frame where the sampled response is closer to isotropic.

>>> import numpy as np
>>> from ge_rbf import IsotropicTransformer, RBFRegressor, gradient_search
>>> from ge_rbf.problems import non_isotropic
>>> X = np.random.default_rng(0).random((30, 2))
>>> y, dy = non_isotropic(X)
>>> frame = IsotropicTransformer(method="ge-lhm").fit(X, dy=dy)
>>> result = gradient_search(
...     RBFRegressor(), frame.transform(X), y, frame.transform_gradient(dy)
... )
>>> model = result.best_estimator
"""

from importlib.metadata import PackageNotFoundError, version

from .kernels import GaussianKernel, Kernel
from .models import RBFRegressor
from .selection import (
    SearchResult,
    gradient_search,
    kfold_search,
    plot_search,
    validation_search,
)
from .transformations import IsotropicTransformer

try:
    __version__ = version("ge_rbf")
except PackageNotFoundError:  # pragma: no cover - only when running from a source tree
    __version__ = "0.0.0+unknown"

__all__ = [
    "GaussianKernel",
    "IsotropicTransformer",
    "Kernel",
    "RBFRegressor",
    "SearchResult",
    "__version__",
    "gradient_search",
    "kfold_search",
    "plot_search",
    "validation_search",
]
