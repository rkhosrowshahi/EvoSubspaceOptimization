"""Factory for building PyMOO algorithm instances from parsed arguments."""

from __future__ import annotations

import numpy as np

from pymoo.core.sampling import Sampling
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.sampling.lhs import LatinHypercubeSampling

from .sampling import GaussianSampling


# ---------------------------------------------------------------------------
# Sampling factory
# ---------------------------------------------------------------------------

def build_sampling(method: str, gaussian_scale: float = 0.5) -> Sampling:
    """Return a PyMOO Sampling object for the given initialization method.

    Supported methods: 'uniform', 'gaussian', 'lhs'.
    """
    method = method.lower()
    if method == "uniform":
        return FloatRandomSampling()
    if method == "gaussian":
        if gaussian_scale <= 0:
            raise ValueError(f"pop_sigma must be > 0 for gaussian init, got {gaussian_scale}")
        return GaussianSampling(scale=gaussian_scale)
    if method == "lhs":
        return LatinHypercubeSampling()
    raise ValueError(
        f"Unknown init_pop method {method!r}. Choose from: uniform, gaussian, lhs."
    )


# ---------------------------------------------------------------------------
# Algorithm factory
# ---------------------------------------------------------------------------

def build_algorithm(args):  # args: argparse.Namespace
    """Construct a PyMOO Algorithm from CLI arguments.

    Supported optimizers: de, pso, es, cmaes.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.  Relevant fields:
        - optimizer         : str
        - pop_size          : int
        - init_pop          : str
        - pop_sigma         : float  (--init_pop gaussian scale)
        - de_mut_rate, de_cr_rate : float  (DE)
        - de_evolving             : bool   (DE: EvolutionaryParameterControl)
        - pso_w, pso_c1, pso_c2   : float  (PSO)
        - pso_evolving            : bool   (PSO: adaptive w/c1/c2)
        - es_sigma          : float  (ES)
        - cmaes_sigma       : float  (CMA-ES)
    """
    optimizer = args.optimizer.lower()
    sampling = build_sampling(args.init_pop, gaussian_scale=args.pop_sigma)

    if optimizer == "de":
        return _build_de(args, sampling)
    if optimizer == "pso":
        return _build_pso(args, sampling)
    if optimizer == "es":
        return _build_es(args, sampling)
    if optimizer == "cmaes":
        return _build_cmaes(args, sampling)

    raise ValueError(
        f"Unknown optimizer {optimizer!r}. Choose from: de, pso, es, cmaes."
    )


# ---------------------------------------------------------------------------
# Individual builders
# ---------------------------------------------------------------------------

def _build_de(args, sampling: Sampling):
    from pymoo.algorithms.soo.nonconvex.de import DE
    from pymoo.operators.control import EvolutionaryParameterControl, NoParameterControl

    # PyMOO passes `control` as a class; Variant instantiates it as control(variant).
    control_cls = (
        EvolutionaryParameterControl if args.de_evolving else NoParameterControl
    )

    # variant=None builds Variant(...) from kwargs; a string variant would ignore
    # selection / n_diffs / crossover passed here.
    return DE(
        pop_size=args.pop_size,
        variant=None,
        CR=args.de_cr_rate,
        F=args.de_mut_rate,
        sampling=sampling,
        selection=args.de_selection,
        n_diffs=args.de_n_diffs,
        crossover=args.de_crossover,
        jitter=args.de_jitter,
        control=control_cls,
    )


def _build_pso(args, sampling: Sampling):
    from pymoo.algorithms.soo.nonconvex.pso import PSO

    return PSO(
        pop_size=args.pop_size,
        w=args.pso_w,
        c1=args.pso_c1,
        c2=args.pso_c2,
        adaptive=args.pso_evolving,
        sampling=sampling,
    )


def _build_es(args, sampling: Sampling):
    """Build a (mu, lambda) Evolution Strategy.

    PyMOO's ES uses self-adaptive sigma via the 1/n rule.
    ``es_sigma`` is used to seed the initial population width via
    a Gaussian sampler (overrides the passed sampler for ES).
    """
    from pymoo.algorithms.soo.nonconvex.es import ES

    # For ES, we use GaussianSampling scaled by es_sigma regardless of
    # the --init_pop flag, since sigma directly controls the initial spread.
    sigma = args.es_sigma
    es_sampling = GaussianSampling(scale=sigma)

    return ES(
        n_offsprings=args.pop_size,
        rule=1.0 / 5.0,    # classic 1/5 success rule
        phi=0.1,
        gamma=0.85,
        sampling=es_sampling,
    )


def _build_cmaes(args, sampling: Sampling):
    from pymoo.algorithms.soo.nonconvex.cmaes import CMAES

    return CMAES(
        sigma=args.cmaes_sigma,
        # CMA-ES manages its own population; pop_size is used as
        # the lambda (offspring count) when explicitly provided.
        restarts=0,
    )
