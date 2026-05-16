"""Custom PyMOO sampling strategies."""

from __future__ import annotations

import numpy as np
from pymoo.core.sampling import Sampling


class GaussianSampling(Sampling):
    """Sample the initial population from a clipped Gaussian distribution.

    Each variable is drawn from N(mu, sigma) where mu is the midpoint of the
    variable bounds and sigma = scale * (upper - lower) / 2.  Samples are
    clipped to [lower, upper].

    Parameters
    ----------
    scale : float
        Controls the width of the distribution relative to the bound range.
        A scale of 0.5 means sigma = 0.5 * half-range (approximately 95 % of
        samples will lie within bounds before clipping).
    """

    def __init__(self, scale: float = 0.5) -> None:
        super().__init__()
        self.scale = scale

    def _do(self, problem, n_samples: int, **kwargs) -> np.ndarray:
        xl, xu = problem.bounds()
        mid = (xl + xu) / 2.0
        sigma = self.scale * (xu - xl) / 2.0
        X = np.random.normal(mid, sigma, size=(n_samples, problem.n_var))
        return np.clip(X, xl, xu)
