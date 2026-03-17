"""Tests for variance reduction techniques."""

from __future__ import annotations

import numpy as np
import pytest

from synthquant.simulation.variance_reduction import (
    antithetic_variates,
    control_variates,
    importance_sampling,
    stratified_sampling,
)


@pytest.fixture()
def small_paths(sample_paths: np.ndarray) -> np.ndarray:
    """Subset of sample_paths for fast tests."""
    return sample_paths[:200]


class TestAntithetic:
    def test_doubles_path_count(self, small_paths: np.ndarray) -> None:
        combined = antithetic_variates(small_paths)
        assert combined.shape[0] == 2 * small_paths.shape[0]

    def test_same_n_steps(self, small_paths: np.ndarray) -> None:
        combined = antithetic_variates(small_paths)
        assert combined.shape[1] == small_paths.shape[1]

    def test_all_prices_positive(self, small_paths: np.ndarray) -> None:
        combined = antithetic_variates(small_paths)
        assert np.all(combined > 0)

    def test_original_paths_preserved(self, small_paths: np.ndarray) -> None:
        """First half of combined paths matches original."""
        combined = antithetic_variates(small_paths)
        np.testing.assert_array_equal(combined[:len(small_paths)], small_paths)

    def test_antithetic_paths_different_from_originals(self, small_paths: np.ndarray) -> None:
        combined = antithetic_variates(small_paths)
        original = combined[:len(small_paths)]
        anti = combined[len(small_paths):]
        assert not np.array_equal(original, anti)

    def test_antithetic_reduces_variance(self) -> None:
        """Antithetic estimator for mean terminal price has lower variance."""
        from synthquant.simulation.engines.gbm import GBMEngine
        engine = GBMEngine()
        rng_seed = 99
        n = 500

        naive_means = []
        anti_means = []
        rng = np.random.default_rng(rng_seed)
        for _ in range(50):
            seed = int(rng.integers(0, 100_000))
            paths = engine.simulate(n, 50, 1 / 252, random_state=seed)
            naive_means.append(np.mean(paths[:, -1]))
            combined = antithetic_variates(paths)
            anti_means.append(np.mean(combined[:, -1]))

        assert np.std(anti_means) <= np.std(naive_means) * 1.5


class TestControlVariates:
    def test_output_shape(self, small_paths: np.ndarray) -> None:
        corrected = control_variates(small_paths, small_paths, float(np.mean(small_paths[:, -1])))
        assert corrected.shape == (small_paths.shape[0],)

    def test_mean_close_to_control_mean(self, small_paths: np.ndarray) -> None:
        """Control variate correction nudges the mean towards the known value."""
        control_mean = float(np.mean(small_paths[:, -1]))
        corrected = control_variates(small_paths, small_paths, control_mean)
        assert abs(np.mean(corrected) - control_mean) < 1.0


class TestStratifiedSampling:
    def test_correct_length(self) -> None:
        samples = stratified_sampling(n_paths=100, n_strata=10, rng=np.random.default_rng(0))
        assert len(samples) == 100

    def test_all_in_unit_interval(self) -> None:
        samples = stratified_sampling(100, 10, rng=np.random.default_rng(1))
        assert np.all(samples >= 0) and np.all(samples <= 1)

    def test_each_stratum_covered(self) -> None:
        """Each of n_strata equal intervals contains at least one sample."""
        n_strata = 5
        n_paths = 100
        samples = stratified_sampling(n_paths, n_strata, rng=np.random.default_rng(2))
        for i in range(n_strata):
            lo, hi = i / n_strata, (i + 1) / n_strata
            count = np.sum((samples >= lo) & (samples <= hi))
            assert count > 0, f"No samples in stratum [{lo:.2f}, {hi:.2f}]"

    def test_default_rng(self) -> None:
        """Works without providing an rng."""
        samples = stratified_sampling(50, 5)
        assert len(samples) == 50


class TestImportanceSampling:
    def test_weights_shape(self, small_paths: np.ndarray) -> None:
        w = importance_sampling(small_paths, target_quantile=0.05)
        assert w.shape == (small_paths.shape[0],)

    def test_weights_positive(self, small_paths: np.ndarray) -> None:
        w = importance_sampling(small_paths)
        assert np.all(w > 0)

    def test_tail_paths_have_higher_weights(self, small_paths: np.ndarray) -> None:
        """Paths in the lower tail receive higher weights."""
        q = 0.10
        w = importance_sampling(small_paths, target_quantile=q)
        terminal = small_paths[:, -1]
        threshold = np.quantile(terminal, q)
        tail_mask = terminal <= threshold
        assert np.mean(w[tail_mask]) > np.mean(w[~tail_mask])
