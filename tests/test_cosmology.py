"""Tests for CosmologyParams CRUD operations through the full macon layer stack."""

from tissage_cosmique.local_sync import cosmology_params


LCDM_PARAMS = dict(
    name="Planck2018",
    Omega_c=0.2589,
    Omega_b=0.0486,
    h=0.6774,
    n_s=0.9667,
    sigma8=0.8159,
    A_s=None,
    Omega_k=0.0,
    w0=-1.0,
    wa=0.0,
)


class TestCosmologyParamsCRUD:
    """Test create, read, filter, update, delete for CosmologyParams."""

    def test_create_and_get(self):
        result = cosmology_params.create_row(**LCDM_PARAMS)
        assert result.name == "Planck2018"
        assert result.Omega_c == 0.2589
        assert result.id_ is not None

        fetched = cosmology_params.get_row(result.id_)
        assert fetched.name == "Planck2018"
        assert fetched.h == 0.6774

    def test_update(self):
        result = cosmology_params.create_row(**LCDM_PARAMS)
        cosmology_params.update_row(result.id_, sigma8=0.82)
        updated = cosmology_params.get_row(result.id_)
        assert updated.sigma8 == 0.82

    def test_delete(self):
        result = cosmology_params.create_row(**LCDM_PARAMS)
        cosmology_params.delete_row(result.id_)
        rows = cosmology_params.get_rows()
        assert len(rows) == 0

    def test_create_multiple_and_get_rows(self):
        cosmology_params.create_row(**LCDM_PARAMS)
        cosmology_params.create_row(
            name="wCDM",
            Omega_c=0.25,
            Omega_b=0.05,
            h=0.70,
            n_s=0.96,
            sigma8=0.80,
            w0=-0.9,
            wa=0.1,
        )
        rows = cosmology_params.get_rows()
        assert len(rows) == 2

    def test_default_values(self):
        result = cosmology_params.create_row(
            name="minimal",
            Omega_c=0.25,
            Omega_b=0.05,
            h=0.70,
            n_s=0.96,
        )
        assert result.Omega_k == 0.0
        assert result.w0 == -1.0
        assert result.wa == 0.0
        assert result.sigma8 is None
        assert result.A_s is None
