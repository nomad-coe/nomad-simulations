from typing import Optional

import pytest

from nomad_simulations.schema_packages.properties import FermiSurface


class TestFermiSurface:
    """
    Test the `FermiSurface` class defined in `properties/band_structure.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    @pytest.mark.parametrize(
        'n_bands, rank',
        [
            (None, None),
            (10, [10]),
        ],
    )
    def test_default_quantities(self, n_bands: Optional[int], rank: Optional[list]):
        """
        Test the default quantities assigned when creating an instance of the `HoppingMatrix` class.
        """
        if n_bands is None:
            with pytest.raises(ValueError) as exc_info:
                fermi_surface = FermiSurface(n_bands=n_bands)
            assert (
                str(exc_info.value)
                == '`n_bands` is not defined during initialization of the class.'
            )
        else:
            fermi_surface = FermiSurface(n_bands=n_bands)
            assert fermi_surface.iri == 'http://fairmat-nfdi.eu/taxonomy/FermiSurface'
            assert fermi_surface.name == 'FermiSurface'
            assert fermi_surface.rank == rank
