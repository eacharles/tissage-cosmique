"""Tests for nuisance parameter evaluation: parametric and functional models."""

import numpy as np
import pytest

from tissage_cosmique.computations.nuisance import evaluate_nuisance


Z = np.linspace(0.0, 2.0, 100)


class TestParametricModels:

    def test_constant(self):
        z_out, f = evaluate_nuisance(Z, model="constant", params={"a0": 1.5})
        np.testing.assert_array_equal(z_out, Z)
        assert np.allclose(f, 1.5)

    def test_linear(self):
        z_out, f = evaluate_nuisance(Z, model="linear", params={"a0": 1.0, "a1": 0.5})
        expected = 1.0 + 0.5 * Z
        np.testing.assert_allclose(f, expected)

    def test_quadratic(self):
        z_out, f = evaluate_nuisance(
            Z, model="quadratic", params={"a0": 1.0, "a1": 0.5, "a2": -0.1},
        )
        expected = 1.0 + 0.5 * Z - 0.1 * Z**2
        np.testing.assert_allclose(f, expected)

    def test_power_law(self):
        z_out, f = evaluate_nuisance(
            Z, model="power_law", params={"a0": 2.0, "a1": 0.8, "z_pivot": 0.5},
        )
        expected = 2.0 * ((1 + Z) / (1 + 0.5)) ** 0.8
        np.testing.assert_allclose(f, expected)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown parametric model"):
            evaluate_nuisance(Z, model="bogus", params={"a0": 1.0})


class TestFunctionalModel:

    def test_interpolation(self):
        nodes_z = [0.0, 0.5, 1.0, 1.5, 2.0]
        nodes_amp = [1.0, 1.2, 1.5, 1.3, 1.1]
        z_out, f = evaluate_nuisance(Z, nodes_z=nodes_z, nodes_amp=nodes_amp)
        np.testing.assert_array_equal(z_out, Z)
        assert f[0] == pytest.approx(1.0, abs=0.01)
        assert f[-1] == pytest.approx(1.1, abs=0.01)
        assert len(f) == len(Z)

    def test_nodes_override_static(self):
        nodes_z = [0.0, 2.0]
        nodes_amp = [3.0, 3.0]
        static_z = [0.0, 2.0]
        static_values = [1.0, 1.0]
        _, f = evaluate_nuisance(
            Z,
            nodes_z=nodes_z, nodes_amp=nodes_amp,
            static_z=static_z, static_values=static_values,
        )
        assert np.allclose(f, 3.0)


class TestStaticArrays:

    def test_static_passthrough(self):
        static_z = [0.0, 1.0, 2.0]
        static_v = [1.5, 1.5, 1.5]
        z_out, f = evaluate_nuisance(Z, static_z=static_z, static_values=static_v)
        np.testing.assert_array_equal(z_out, [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(f, [1.5, 1.5, 1.5])


class TestPriority:

    def test_parametric_overrides_functional_and_static(self):
        _, f = evaluate_nuisance(
            Z,
            model="constant", params={"a0": 5.0},
            nodes_z=[0.0, 2.0], nodes_amp=[1.0, 1.0],
            static_z=[0.0, 2.0], static_values=[2.0, 2.0],
        )
        assert np.allclose(f, 5.0)

    def test_functional_overrides_static(self):
        _, f = evaluate_nuisance(
            Z,
            nodes_z=[0.0, 2.0], nodes_amp=[7.0, 7.0],
            static_z=[0.0, 2.0], static_values=[2.0, 2.0],
        )
        assert np.allclose(f, 7.0)

    def test_none_when_nothing_provided(self):
        result = evaluate_nuisance(Z)
        assert result is None


class TestIntegrationWithTracers:

    def test_parametric_bias_in_tracer(self):
        pytest.importorskip("pyccl")
        from tissage_cosmique.computations.cosmology import make_pyccl_cosmology
        from tissage_cosmique.computations.tracers import make_pyccl_tracer

        cosmo = make_pyccl_cosmology(dict(
            Omega_c=0.25, Omega_b=0.05, h=0.70, n_s=0.96, sigma8=0.81,
        ))
        config = dict(
            tracer_type="number_counts",
            dndz_z=Z.tolist(),
            dndz_nz=np.exp(-0.5 * ((Z - 0.5) / 0.15)**2).tolist(),
            bias_model="linear",
            bias_params={"a0": 1.5, "a1": 0.3},
            has_rsd=False,
        )
        tracer = make_pyccl_tracer(cosmo, config)
        assert tracer is not None

    def test_functional_bias_in_tracer(self):
        pytest.importorskip("pyccl")
        from tissage_cosmique.computations.cosmology import make_pyccl_cosmology
        from tissage_cosmique.computations.tracers import make_pyccl_tracer

        cosmo = make_pyccl_cosmology(dict(
            Omega_c=0.25, Omega_b=0.05, h=0.70, n_s=0.96, sigma8=0.81,
        ))
        config = dict(
            tracer_type="number_counts",
            dndz_z=Z.tolist(),
            dndz_nz=np.exp(-0.5 * ((Z - 0.5) / 0.15)**2).tolist(),
            bias_nodes_z=[0.0, 0.5, 1.0, 1.5, 2.0],
            bias_nodes_amp=[1.2, 1.4, 1.6, 1.5, 1.3],
            has_rsd=False,
        )
        tracer = make_pyccl_tracer(cosmo, config)
        assert tracer is not None
