"""Model calibration: MLE and Bayesian approaches."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["Calibrator", "MLECalibrator", "BayesianCalibrator", "CalibratedParams"]


@dataclass
class CalibratedParams:
    """Container for calibrated model parameters.

    Attributes:
        params: Dict of parameter name to fitted value.
        method: Calibration method used ('MLE' or 'Bayesian').
        log_likelihood: Log-likelihood at fitted parameters (MLE only).
        convergence_info: Optimizer convergence details (MLE only).
        posterior_samples: Posterior samples dict (Bayesian only).
    """

    params: dict[str, float]
    method: str
    log_likelihood: float | None = None
    convergence_info: dict[str, Any] = field(default_factory=dict)
    posterior_samples: dict[str, np.ndarray] | None = None


class Calibrator(ABC):
    """Abstract base class for model calibration."""

    @abstractmethod
    def calibrate(
        self,
        log_likelihood_fn: Callable[..., float],
        initial_params: dict[str, float],
        bounds: dict[str, tuple[float, float]] | None = None,
    ) -> CalibratedParams:
        """Calibrate model parameters.

        Args:
            log_likelihood_fn: Function that takes a dict of params and returns log-likelihood.
            initial_params: Starting parameter values.
            bounds: Optional bounds dict mapping param name to (lower, upper).

        Returns:
            CalibratedParams with fitted parameter values.
        """
        ...


class MLECalibrator(Calibrator):
    """Maximum Likelihood Estimation calibrator using scipy.optimize.

    Args:
        method: Optimization method passed to scipy.optimize.minimize.
        options: Additional options for scipy.optimize.minimize.
    """

    def __init__(
        self,
        method: str = "L-BFGS-B",
        options: dict[str, Any] | None = None,
    ) -> None:
        self.method = method
        self.options = options or {"maxiter": 1000, "ftol": 1e-9}

    def calibrate(
        self,
        log_likelihood_fn: Callable[..., float],
        initial_params: dict[str, float],
        bounds: dict[str, tuple[float, float]] | None = None,
    ) -> CalibratedParams:
        """Calibrate via MLE.

        Args:
            log_likelihood_fn: Callable mapping param dict -> log-likelihood.
            initial_params: Initial parameter guess.
            bounds: Parameter bounds.

        Returns:
            CalibratedParams with fitted values and convergence info.
        """
        from scipy.optimize import minimize

        param_names = list(initial_params.keys())
        x0 = np.array([initial_params[k] for k in param_names])
        scipy_bounds = None
        if bounds:
            scipy_bounds = [bounds.get(k, (None, None)) for k in param_names]

        def neg_ll(x: np.ndarray) -> float:
            p = dict(zip(param_names, x, strict=False))
            return -log_likelihood_fn(**p)

        result = minimize(
            neg_ll,
            x0,
            method=self.method,
            bounds=scipy_bounds,
            options=self.options,
        )

        fitted = dict(zip(param_names, result.x, strict=False))
        logger.info(f"MLE calibration: success={result.success}, nll={result.fun:.6f}")
        return CalibratedParams(
            params=fitted,
            method="MLE",
            log_likelihood=-result.fun,
            convergence_info={
                "success": result.success,
                "message": result.message,
                "n_eval": result.nfev,
            },
        )


class BayesianCalibrator(Calibrator):
    """Bayesian calibrator using PyMC (optional dependency).

    Args:
        n_draws: Number of posterior samples per chain.
        n_chains: Number of MCMC chains.
        target_accept: Target acceptance rate for NUTS sampler.
    """

    def __init__(
        self,
        n_draws: int = 1000,
        n_chains: int = 2,
        target_accept: float = 0.9,
    ) -> None:
        self.n_draws = n_draws
        self.n_chains = n_chains
        self.target_accept = target_accept

    def calibrate(
        self,
        log_likelihood_fn: Callable[..., float],
        initial_params: dict[str, float],
        bounds: dict[str, tuple[float, float]] | None = None,
    ) -> CalibratedParams:
        """Calibrate via PyMC MCMC.

        Args:
            log_likelihood_fn: Callable mapping param dict -> log-likelihood.
            initial_params: Initial parameter guess (used as starting point).
            bounds: Parameter bounds (used to define Uniform priors when provided).

        Returns:
            CalibratedParams with posterior means and posterior_samples dict.

        Raises:
            ImportError: If PyMC is not installed.
        """
        try:
            import pymc as pm
            import pytensor.tensor as pt
        except ImportError as e:
            raise ImportError(
                "PyMC is required for BayesianCalibrator. "
                "Install with: pip install synthquant[bayesian]"
            ) from e

        param_names = list(initial_params.keys())
        _bounds = bounds or {}

        with pm.Model() as model:  # noqa: F841
            priors = {}
            for name in param_names:
                lo, hi = _bounds.get(name, (-np.inf, np.inf))
                if np.isfinite(lo) and np.isfinite(hi):
                    priors[name] = pm.Uniform(name, lower=lo, upper=hi)
                else:
                    priors[name] = pm.Normal(
                        name, mu=initial_params[name], sigma=abs(initial_params[name]) + 1.0
                    )

            # Custom log-likelihood potential
            @pm.potential
            def ll_potential() -> Any:
                p = {k: priors[k] for k in param_names}
                return pt.as_tensor_variable(log_likelihood_fn(**p))

            trace = pm.sample(
                draws=self.n_draws,
                chains=self.n_chains,
                target_accept=self.target_accept,
                progressbar=False,
                return_inferencedata=True,
            )

        posterior = {
            k: trace.posterior[k].values.flatten() for k in param_names
        }
        fitted = {k: float(np.mean(posterior[k])) for k in param_names}
        logger.info(f"Bayesian calibration complete: {self.n_draws} draws x {self.n_chains} chains")
        return CalibratedParams(
            params=fitted,
            method="Bayesian",
            posterior_samples=posterior,
        )
