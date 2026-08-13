r"""Making trajectory data usable by a method that assumes scattered samples.

A continuation or transient solver does not return scattered points. It returns a handful
of **trajectories**: one per design, each a chain of closely spaced states parametrised by
a time-like coordinate — arc length along a load path, time in a transient, a continuation
parameter. Every stored point carries a gradient in all inputs, the trajectory coordinate
included, so a single solve contributes as many rows as it took steps.

Stacked into one ``(n_samples, n_features)`` block that is a very particular clustering,
and it breaks the usual defaults. Normalise every input to :math:`[0, 1]` and the samples
lie on a few long thin curves: neighbouring designs might be :math:`1/3` apart while
consecutive points on one trajectory are :math:`1/17` apart. Every point's nearest
neighbours are then the points before and after it *on its own trajectory*, and nothing
else.

That breaks curvature estimation specifically. :class:`~ge_rbf.transformations.IsotropicTransformer`
with ``method="ge-lhm"`` builds each local Hessian from ``n_features + 1`` neighbours, so
in two dimensions it uses exactly three — and all three are collinear. Measured on real
load-path data, **100% of every point's neighbours were on its own trajectory**: the
curvature across designs was estimated from no spread whatever. The symptom is an
anisotropy ratio inflated by one or two orders of magnitude, and sometimes a curvature
estimate so far from positive definite that no frame exists at all.

:class:`TrajectoryScaler` fixes it by stretching the trajectory axis until one step along a
trajectory is ``stretch`` times the distance between neighbouring designs, which makes a
point's nearest neighbours the adjacent *designs* at a similar point along their own
trajectories — which is what a curvature estimate needs to see.

Two properties worth knowing before reaching for something more elaborate:

**It is unit-free.** The factor is a ratio of a design-space distance to a trajectory-space
distance, so it already carries the units of one per the other. Rescaling the trajectory
coordinate — seconds to milliseconds, millimetres to metres — cancels exactly, and no
normalisation of that axis is needed first. The *design* columns do need to be on
comparable scales with each other, since a Euclidean distance across mixed units is
dominated by whichever has the largest range.

**It is a diagonal frame.** The four methods below have the same shape as
:class:`~ge_rbf.transformations.IsotropicTransformer`'s, so the two chain directly, and
what this class really is is a diagonal ``method="ideal"`` frame whose one non-unit entry
is estimated from the sample layout rather than supplied.

Typical use, with the frame estimated *after* the scaling because that is the whole point:

.. code-block:: python

    scaler = TrajectoryScaler().fit(Z, groups=groups)
    Zs, dys = scaler.transform(Z), scaler.transform_gradient(dy)

    frame = IsotropicTransformer(fallback=True).fit(Zs, dy=dys)
    Zt, dyt = frame.transform(Zs), frame.transform_gradient(dys)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

from ._linalg import check_samples

__all__ = ["TrajectoryScaler"]


class TrajectoryScaler(TransformerMixin, BaseEstimator):
    """Rescale a trajectory coordinate so it is commensurate with the design spacing.

    Parameters
    ----------
    trajectory_column : int, optional
        Which column holds the trajectory coordinate. Negative indices count from the end,
        and the default ``-1`` matches the layout produced by
        :func:`ge_rbf.problems.load_path_samples`: design variables first, trajectory
        coordinate last.
    stretch : float, optional
        **A target ratio, not a magnification.** After the transform, one step along a
        trajectory is ``stretch`` times the nearest-neighbour distance between distinct
        designs. The default of ``1.5`` puts a step just beyond the design spacing, so the
        adjacent designs win the neighbour search but the trajectory ordering is not
        destroyed. ``scale_`` comes out below 1 when the trajectory axis started out
        coarser than the designs, which is correct.

    Attributes
    ----------
    scale_ : float
        The factor the trajectory axis was multiplied by. The first thing to check.
    scaling_ : ndarray of shape (n_features,)
        Ones, with ``scale_`` at ``trajectory_column``. This is what the transforms apply.
    design_spacing_ : float
        Mean nearest-neighbour distance between trajectories, measured between their
        representative points in the design columns.
    step_spacing_ : float
        Mean distance between consecutive points within a trajectory.
    groups_ : ndarray of shape (n_samples,)
        The trajectory labels actually used, whether supplied or inferred.
    n_trajectories_ : int
    trajectory_column_ : int
        ``trajectory_column`` resolved to a non-negative index.
    n_features_in_ : int

    Examples
    --------
    >>> from ge_rbf import TrajectoryScaler
    >>> from ge_rbf.problems import load_path, load_path_samples
    >>> Z, groups = load_path_samples()
    >>> y, dy = load_path(Z)
    >>> scaler = TrajectoryScaler().fit(Z, groups=groups)
    >>> Zs, dys = scaler.transform(Z), scaler.transform_gradient(dy)
    """

    def __init__(self, trajectory_column: int = -1, stretch: float = 1.5) -> None:
        self.trajectory_column = trajectory_column
        self.stretch = stretch

    # ------------------------------------------------------------------ fitting

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        dy: ArrayLike | None = None,
        groups: ArrayLike | None = None,
    ) -> TrajectoryScaler:
        """Estimate the trajectory scale factor from the sample layout.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            One row per stored point, design variables and trajectory coordinate together.
        y, dy : optional
            Accepted and **ignored**. The scale factor is purely geometric — it depends on
            where the samples are, not on what was measured there. They are in the
            signature so that ``fit_transform`` works and so that a caller can fit this and
            an :class:`~ge_rbf.transformations.IsotropicTransformer` with the same call.
        groups : array_like of shape (n_samples,), optional
            Which trajectory each row belongs to. Any labels that compare equal will do::

                groups = np.repeat(np.arange(n_designs), n_steps)          # equal lengths
                groups = np.concatenate([np.full(p.size, i) for ...])      # ragged

            ``None`` infers the grouping from the design columns, which requires the design
            row to repeat **bit-exactly** down a trajectory. That holds if the block was
            built with ``np.repeat`` and fails if each row was stored with its own
            round-off, so pass ``groups`` explicitly whenever the data came back from a
            solver.

        Returns
        -------
        self
        """
        X = check_samples(X)
        n_samples, n_features = X.shape

        if n_features < 2:
            raise ValueError(
                f"X needs at least one design column and a trajectory column, so at least "
                f"2 features; got {n_features}."
            )

        column = int(self.trajectory_column)
        if not -n_features <= column < n_features:
            raise ValueError(
                f"trajectory_column={self.trajectory_column} is out of range for "
                f"{n_features} features."
            )
        column %= n_features

        stretch = float(self.stretch)
        if not np.isfinite(stretch) or stretch <= 0:
            raise ValueError(f"stretch must be finite and positive, got {self.stretch!r}.")

        design_columns = np.delete(X, column, axis=1)
        labels = self._resolve_groups(groups, design_columns, n_samples)
        unique = np.unique(labels)

        if unique.size < 2:
            raise ValueError(
                "At least 2 trajectories are needed: the scale factor balances the spacing "
                "along a trajectory against the spacing between them, and with one "
                "trajectory there is nothing to balance against. Fit on data from more "
                "than one design, or leave the coordinate unscaled."
            )

        # One representative point per trajectory. Taking the mean rather than requiring
        # an exactly repeated row means a design perturbed by solver round-off still gives
        # a sensible spacing.
        representatives = np.array(
            [design_columns[labels == label].mean(axis=0) for label in unique]
        )
        between = cdist(representatives, representatives)
        np.fill_diagonal(between, np.inf)
        design_spacing = float(np.mean(between.min(axis=1)))

        if design_spacing <= 0:
            raise ValueError(
                "Two trajectories sit at the same design point, so the spacing between "
                "designs is zero and there is nothing to scale the trajectory axis "
                "against. Merge repeated runs of one design into a single group, or drop "
                "the duplicates."
            )

        steps = [np.diff(np.sort(X[labels == label, column])) for label in unique]
        steps = np.concatenate([s for s in steps if s.size] or [np.zeros(0)])

        if steps.size == 0:
            raise ValueError(
                "Every trajectory holds a single point, so there is no step along one to "
                "measure. Trajectory data means several points per design."
            )

        step_spacing = float(np.mean(steps))
        if step_spacing <= 0:
            raise ValueError(
                "The mean step along a trajectory is zero, so the trajectory coordinate "
                "does not vary within a trajectory. Check that groups identifies "
                "trajectories and not individual points."
            )

        scaling = np.ones(n_features)
        scaling[column] = stretch * design_spacing / step_spacing

        self.trajectory_column_ = column
        self.design_spacing_ = design_spacing
        self.step_spacing_ = step_spacing
        self.scale_ = float(scaling[column])
        self.scaling_ = scaling
        self.groups_ = labels
        self.n_trajectories_ = int(unique.size)
        self.n_features_in_ = n_features

        return self

    @staticmethod
    def _resolve_groups(
        groups: ArrayLike | None, design_columns: NDArray[np.float64], n_samples: int
    ) -> NDArray:
        """Trajectory labels, either as supplied or inferred from the design columns."""
        if groups is not None:
            labels = np.asarray(groups).ravel()
            if labels.shape[0] != n_samples:
                raise ValueError(
                    f"groups has {labels.shape[0]} entries but there are {n_samples} samples."
                )
            return labels

        _, labels = np.unique(design_columns, axis=0, return_inverse=True)
        labels = labels.ravel()

        if labels.size == np.unique(labels).size:
            raise ValueError(
                "No two rows share a design, so no trajectory could be inferred. Inference "
                "compares design rows exactly, which fails when each row carries its own "
                "round-off; pass groups explicitly."
            )

        return labels

    # ------------------------------------------------------------ applying the scale

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Stretch the trajectory coordinate: ``X * scaling_``."""
        X = self._check_applied_to(X, "X")
        return X * self.scaling_

    def inverse_transform(self, Xt: ArrayLike) -> NDArray[np.float64]:
        """Map stretched samples back to the original coordinates."""
        Xt = self._check_applied_to(Xt, "Xt")
        return Xt / self.scaling_

    def transform_gradient(self, dy: ArrayLike) -> NDArray[np.float64]:
        r"""Express gradients in the stretched coordinates.

        Stretching an axis rescales the derivative along it: with
        :math:`\hat{s} = \sigma s`, the chain rule gives
        :math:`\partial f / \partial \hat{s} = \sigma^{-1} \partial f / \partial s`, so
        this **divides** where :meth:`transform` multiplies.

        Skipping this conversion is the quiet failure mode. The function values stay right,
        the fit is handed gradients belonging to a different coordinate system, and nothing
        raises.
        """
        dy = self._check_applied_to(dy, "dy")
        return dy / self.scaling_

    def inverse_transform_gradient(self, dyt: ArrayLike) -> NDArray[np.float64]:
        """Map gradients in the stretched coordinates back to the original ones."""
        dyt = self._check_applied_to(dyt, "dyt")
        return dyt * self.scaling_

    def _check_applied_to(self, array: ArrayLike, name: str) -> NDArray[np.float64]:
        if not hasattr(self, "scaling_"):
            raise NotFittedError(f"This {type(self).__name__} is not fitted yet. Call fit first.")

        array = check_samples(array, name=name)
        if array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"{name} has {array.shape[1]} features, but this transformer was fitted "
                f"on {self.n_features_in_}."
            )
        return array
