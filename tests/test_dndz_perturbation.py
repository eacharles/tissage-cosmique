"""Tests for n(z) perturbation models: shift, shift+stretch, comb."""

import pytest

import numpy as np

from tissage_cosmique.computations.nuisance import perturb_dndz


Z = np.linspace(0.0, 3.0, 300)
NZ = np.exp(-0.5 * ((Z - 0.8) / 0.2)**2)


def _peak_z(z: np.ndarray, nz: np.ndarray) -> float:
    return float(z[np.argmax(nz)])


def _width(z: np.ndarray, nz: np.ndarray) -> float:
    nz_norm = nz / nz.max()
    half = nz_norm > 0.5
    return float(z[half][-1] - z[half][0])


class TestShift:

    def test_positive_shift(self):
        _, nz_new = perturb_dndz(Z, NZ, model="shift", params={"dz": 0.1})
        assert _peak_z(Z, nz_new) == pytest.approx(0.9, abs=0.02)

    def test_negative_shift(self):
        _, nz_new = perturb_dndz(Z, NZ, model="shift", params={"dz": -0.1})
        assert _peak_z(Z, nz_new) == pytest.approx(0.7, abs=0.02)

    def test_zero_shift_preserves(self):
        _, nz_new = perturb_dndz(Z, NZ, model="shift", params={"dz": 0.0})
        np.testing.assert_allclose(nz_new, NZ, atol=1e-10)

    def test_normalization_preserved(self):
        _, nz_new = perturb_dndz(Z, NZ, model="shift", params={"dz": 0.2})
        np.testing.assert_allclose(
            np.trapezoid(nz_new, Z), np.trapezoid(NZ, Z), rtol=1e-3,
        )


class TestShiftStretch:

    def test_shift_moves_peak(self):
        _, nz_new = perturb_dndz(
            Z, NZ, model="shift_stretch", params={"dz": 0.1, "sigma_z": 1.0},
        )
        assert _peak_z(Z, nz_new) > _peak_z(Z, NZ)

    def test_stretch_widens(self):
        _, nz_wide = perturb_dndz(
            Z, NZ, model="shift_stretch", params={"dz": 0.0, "sigma_z": 1.5},
        )
        assert _width(Z, nz_wide) > _width(Z, NZ) * 1.2

    def test_stretch_narrows(self):
        _, nz_narrow = perturb_dndz(
            Z, NZ, model="shift_stretch", params={"dz": 0.0, "sigma_z": 0.7},
        )
        assert _width(Z, nz_narrow) < _width(Z, NZ) * 0.9

    def test_identity(self):
        _, nz_new = perturb_dndz(
            Z, NZ, model="shift_stretch", params={"dz": 0.0, "sigma_z": 1.0},
        )
        np.testing.assert_allclose(nz_new, NZ, atol=1e-6)

    def test_normalization_preserved(self):
        _, nz_new = perturb_dndz(
            Z, NZ, model="shift_stretch",
            params={"dz": 0.1, "sigma_z": 1.3},
        )
        np.testing.assert_allclose(
            np.trapezoid(nz_new, Z), np.trapezoid(NZ, Z), rtol=1e-3,
        )


class TestComb:

    def test_positive_perturbation(self):
        _, nz_new = perturb_dndz(
            Z, NZ, model="comb",
            basis_z=[0.0, 0.5, 1.0, 1.5, 2.0],
            basis_amp=[0.0, 0.1, 0.0, 0.0, 0.0],
        )
        mask = (Z > 0.4) & (Z < 0.6)
        assert np.mean(nz_new[mask]) > np.mean(NZ[mask])

    def test_zero_perturbation(self):
        _, nz_new = perturb_dndz(
            Z, NZ, model="comb",
            basis_z=[0.0, 1.0, 2.0],
            basis_amp=[0.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(nz_new, NZ, rtol=1e-3)

    def test_normalization_preserved(self):
        _, nz_new = perturb_dndz(
            Z, NZ, model="comb",
            basis_z=[0.0, 0.5, 1.0, 1.5, 2.0],
            basis_amp=[0.05, -0.1, 0.05, 0.02, 0.0],
        )
        np.testing.assert_allclose(
            np.trapezoid(nz_new, Z), np.trapezoid(NZ, Z), rtol=1e-3,
        )

    def test_no_basis_returns_original(self):
        _, nz_new = perturb_dndz(Z, NZ, model="comb")
        np.testing.assert_array_equal(nz_new, NZ)


class TestNoModel:

    def test_none_returns_original(self):
        z_out, nz_out = perturb_dndz(Z, NZ)
        np.testing.assert_array_equal(z_out, Z)
        np.testing.assert_array_equal(nz_out, NZ)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown dndz"):
            perturb_dndz(Z, NZ, model="bogus", params={})


class TestIntegrationWithTracer:

    def test_shifted_tracer_changes_cl(self):
        pytest.importorskip("pyccl")
        from tissage_cosmique.computations.angular_cl import angular_cl

        cosmo_params = dict(
            Omega_c=0.25, Omega_b=0.05, h=0.70,
            n_s=0.96, sigma8=0.81,
        )
        ell = np.arange(10, 200).astype(float)

        base_config = dict(
            tracer_type="weak_lensing",
            dndz_z=Z.tolist(), dndz_nz=NZ.tolist(),
            has_shear=True,
        )
        shifted_config = dict(
            tracer_type="weak_lensing",
            dndz_z=Z.tolist(), dndz_nz=NZ.tolist(),
            dndz_model="shift", dndz_params={"dz": 0.3},
            has_shear=True,
        )

        cl_base = angular_cl(cosmo_params, base_config, base_config, ell)
        cl_shifted = angular_cl(
            cosmo_params, shifted_config, shifted_config, ell,
        )
        assert not np.allclose(cl_base, cl_shifted, rtol=0.01)
