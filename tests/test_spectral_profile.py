#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pytest
import numpy as np
from typing import Optional, List, Union

from . import logger

from nomad.units import ureg
from nomad_simulations.outputs import Outputs
from nomad_simulations.properties import (
    SpectralProfile,
    ElectronicDensityOfStates,
    XASSpectra,
    FermiLevel,
)
from nomad_simulations.variables import Temperature, Energy2 as Energy


class TestSpectralProfile:
    """
    Test the `SpectralProfile` class defined in `properties/spectral_profile.py`.
    """

    def test_is_valid_spectral_profile(self):
        """
        Test the `is_valid_spectral_profile` method.
        """
        spectral_profile = SpectralProfile(
            variables=[Energy(grid_points=[-3, -2, -1, 0, 1, 2, 3] * ureg.joule)]
        )
        spectral_profile.value = [1.5, 1.2, 0, 0, 0, 0.8, 1.3]
        assert spectral_profile.is_valid_spectral_profile()
        spectral_profile.value = [3, 2, 0, 0, 0, -4, 1]
        assert not spectral_profile.is_valid_spectral_profile()
        # default value inherited in other SpectralProfile classes
        assert spectral_profile.rank == []


class TestElectronicDensityOfStates:
    """
    Test the `ElectronicDensityOfStates` class defined in `properties/spectral_profile.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `ElectronicDensityOfStates` class.
        """
        electronic_dos = ElectronicDensityOfStates()
        assert (
            electronic_dos.iri
            == 'http://fairmat-nfdi.eu/taxonomy/ElectronicDensityOfStates'
        )
        assert electronic_dos.name == 'ElectronicDensityOfStates'
        assert electronic_dos.rank == []

    def test_check_energy_variables(self):
        """
        Test the `_check_energy_variables` method.
        """
        electronic_dos = ElectronicDensityOfStates()
        electronic_dos.variables = [
            Temperature(grid_points=[-3, -2, -1, 0, 1, 2, 3] * ureg.kelvin)
        ]
        assert electronic_dos._check_energy_variables(logger) is None
        electronic_dos.variables.append(
            Energy(grid_points=[-3, -2, -1, 0, 1, 2, 3] * ureg.joule)
        )
        energies = electronic_dos._check_energy_variables(logger)
        assert (energies.magnitude == np.array([-3, -2, -1, 0, 1, 2, 3])).all()

    @pytest.mark.parametrize(
        'fermi_level, sibling_section_value, result',
        [
            (None, None, None),
            (None, 0.5, 0.5),
            (0.5, None, 0.5),
            (0.5, 1.0, 0.5),
        ],
    )
    def test_resolve_fermi_level(
        self,
        fermi_level: Optional[float],
        sibling_section_value: Optional[float],
        result: Optional[float],
    ):
        """
        Test the `_resolve_fermi_level` method.
        """
        outputs = Outputs()
        sec_fermi_level = FermiLevel(variables=[])
        if sibling_section_value is not None:
            sec_fermi_level.value = sibling_section_value * ureg.joule
        outputs.fermi_level.append(sec_fermi_level)
        electronic_dos = ElectronicDensityOfStates(
            variables=[Energy(grid_points=[-3, -2, -1, 0, 1, 2, 3] * ureg.joule)]
        )
        electronic_dos.value = np.array([1.5, 1.2, 0, 0, 0, 0.8, 1.3]) * ureg('1/joule')
        if fermi_level is not None:
            electronic_dos.fermi_level = fermi_level * ureg.joule
        outputs.electronic_dos.append(electronic_dos)
        resolved_fermi_level = electronic_dos.resolve_fermi_level(logger)
        if resolved_fermi_level is not None:
            resolved_fermi_level = resolved_fermi_level.magnitude
        assert resolved_fermi_level == result


class TestXASSpectra:
    """
    Test the `XASSpectra` class defined in `properties/spectral_profile.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `XASSpectra` class.
        """
        xas_spectra = XASSpectra()
        assert xas_spectra.iri is None  # Add iri when available
        assert xas_spectra.name == 'XASSpectra'
        assert xas_spectra.rank == []
