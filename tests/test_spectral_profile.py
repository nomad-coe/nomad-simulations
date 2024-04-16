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
from nomad_simulations.properties import (
    SpectralProfile,
    ElectronicDensityOfStates,
    XASSpectra,
)
from nomad_simulations.variables import Temperature


class TestSpectralProfile:
    """
    Test the `SpectralProfile` class defined in `properties/spectral_profile.py`.
    """

    def test_negative_value(self):
        spectral_profile = SpectralProfile()
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
        electronic_band_gap = ElectronicDensityOfStates()
        assert (
            electronic_band_gap.iri
            == 'http://fairmat-nfdi.eu/taxonomy/ElectronicDensityOfStates'
        )
        assert electronic_band_gap.name == 'ElectronicDensityOfStates'
        assert electronic_band_gap.rank == []


class TestXASSpectra:
    """
    Test the `XASSpectra` class defined in `properties/spectral_profile.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `XASSpectra` class.
        """
        electronic_band_gap = XASSpectra()
        assert electronic_band_gap.iri is None  # Add iri when available
        assert electronic_band_gap.name == 'XASSpectra'
        assert electronic_band_gap.rank == []
