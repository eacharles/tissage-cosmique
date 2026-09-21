"""Tests for ℓ-binning utilities."""

import numpy as np
import pytest

from tissage_cosmique.computations.ell_binning import (
    bin_cl,
    make_linear_bins,
    make_log_bins,
    pair_key,
)


class TestBinCl:

    def test_uniform_weights(self):
        ell = np.arange(10, 110).astype(float)
        cl = np.ones(100)
        edges = [10, 60, 110]
        ell_eff, cl_binned = bin_cl(ell, cl, edges, weights="uniform")
        assert len(ell_eff) == 2
        assert len(cl_binned) == 2
        np.testing.assert_allclose(cl_binned, [1.0, 1.0])

    def test_2ell_plus_1_weights(self):
        ell = np.arange(10, 110).astype(float)
        cl = np.ones(100)
        edges = [10, 60, 110]
        ell_eff, cl_binned = bin_cl(ell, cl, edges, weights="2ell+1")
        assert len(cl_binned) == 2
        np.testing.assert_allclose(cl_binned, [1.0, 1.0])
        assert ell_eff[1] > ell_eff[0]

    def test_weighted_mean_value(self):
        ell = np.array([10.0, 11.0, 12.0])
        cl = np.array([1.0, 2.0, 3.0])
        edges = [10, 13]
        _, cl_binned = bin_cl(ell, cl, edges, weights="uniform")
        assert cl_binned[0] == pytest.approx(2.0)

    def test_empty_bin(self):
        ell = np.array([10.0, 11.0])
        cl = np.array([1.0, 2.0])
        edges = [10, 12, 100]
        _, cl_binned = bin_cl(ell, cl, edges, weights="uniform")
        assert cl_binned[1] == 0.0

    def test_unknown_weights_raises(self):
        with pytest.raises(ValueError, match="Unknown weights"):
            bin_cl(np.array([10.0]), np.array([1.0]), [10, 20], weights="bad")

    def test_preserves_total_power(self):
        ell = np.arange(10, 1000).astype(float)
        cl = 1.0 / ell**2
        edges = make_log_bins(10, 1000, 10)
        _, cl_binned = bin_cl(ell, cl, edges, weights="uniform")
        assert len(cl_binned) == 10
        assert all(cb > 0 for cb in cl_binned)


class TestMakeBins:

    def test_log_bins_shape(self):
        edges = make_log_bins(10, 1000, 20)
        assert len(edges) == 21
        assert edges[0] == pytest.approx(10)
        assert edges[-1] == pytest.approx(1000)

    def test_log_bins_spacing(self):
        edges = make_log_bins(10, 1000, 5)
        ratios = edges[1:] / edges[:-1]
        np.testing.assert_allclose(ratios, ratios[0], rtol=1e-10)

    def test_linear_bins_shape(self):
        edges = make_linear_bins(10, 1000, 20)
        assert len(edges) == 21
        assert edges[0] == pytest.approx(10)
        assert edges[-1] == pytest.approx(1000)

    def test_linear_bins_spacing(self):
        edges = make_linear_bins(10, 1000, 5)
        diffs = np.diff(edges)
        np.testing.assert_allclose(diffs, diffs[0], rtol=1e-10)


class TestPairKey:

    def test_pair_key(self):
        assert pair_key("wl", "nc") == "wl-nc"
        assert pair_key("a", "b") == "a-b"
