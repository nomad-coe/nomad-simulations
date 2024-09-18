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

from typing import Optional, Union

import numpy as np
import pytest
from nomad.datamodel import EntryArchive
from nomad.units import ureg

from nomad_simulations.schema_packages.properties import ElectronicBandGap
from nomad_simulations.schema_packages.variables import Temperature

from . import logger


class TestElectronicBandGap:
    """
    Test the `ElectronicBandGap` class defined in `properties/band_gap.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `ElectronicBandGap` class.
        """
        electronic_band_gap = ElectronicBandGap()
        assert (
            electronic_band_gap.iri
            == 'http://fairmat-nfdi.eu/taxonomy/ElectronicBandGap'
        )
        assert electronic_band_gap.name == 'ElectronicBandGap'
        assert electronic_band_gap.rank == []

    @pytest.mark.parametrize(
        'value, result',
        [
            (0.0, 0.0),
            (1.0, 1.0),
            (-1.0, None),
            ([1.0, 2.0, -1.0], None),
        ],
    )
    def test_validate_values(self, value: Union[list[float], float], result: float):
        """
        Test the `validate_values` method.
        """
        if isinstance(value, list):
            electronic_band_gap = ElectronicBandGap(
                variables=[Temperature(points=[1, 2, 3] * ureg.kelvin)]
            )
        else:
            electronic_band_gap = ElectronicBandGap()
        electronic_band_gap.value = value * ureg.joule
        validated_value = electronic_band_gap.validate_values(logger)
        if validated_value is not None:
            assert np.isclose(validated_value.magnitude, result)
        else:
            assert validated_value == result

    @pytest.mark.parametrize(
        'momentum_transfer, type, result',
        [
            (None, None, None),
            (None, 'direct', 'direct'),
            (None, 'indirect', 'indirect'),
            ([[0, 0, 0]], None, None),
            ([[0, 0, 0]], 'direct', None),
            ([[0, 0, 0]], 'indirect', None),
            ([[0, 0, 0], [0, 0, 0]], None, 'direct'),
            ([[0, 0, 0], [0, 0, 0]], 'direct', 'direct'),
            ([[0, 0, 0], [0, 0, 0]], 'indirect', 'direct'),
            ([[0, 0, 0], [0.5, 0.5, 0.5]], None, 'indirect'),
            ([[0, 0, 0], [0.5, 0.5, 0.5]], 'direct', 'indirect'),
            ([[0, 0, 0], [0.5, 0.5, 0.5]], 'indirect', 'indirect'),
        ],
    )
    def test_resolve_type(
        self, momentum_transfer: Optional[list[float]], type: str, result: Optional[str]
    ):
        """
        Test the `resolve_type` method.
        """
        electronic_band_gap = ElectronicBandGap(
            variables=[],
            momentum_transfer=momentum_transfer,
            type=type,
        )
        assert electronic_band_gap.resolve_type(logger) == result

    def test_normalize(self):
        """
        Test the `normalize` method for two different ElectronicBandGap instantiations, one with a scalar
        `value` and another with a temperature-dependent `value`
        """
        scalar_band_gap = ElectronicBandGap(variables=[], type='direct')
        scalar_band_gap.value = 1.0 * ureg.joule
        scalar_band_gap.normalize(EntryArchive(), logger)
        assert scalar_band_gap.type == 'direct'
        assert np.isclose(scalar_band_gap.value.magnitude, 1.0)

        t_dependent_band_gap = ElectronicBandGap(
            variables=[Temperature(points=[0, 10, 20, 30] * ureg.kelvin)],
            type='direct',
        )
        t_dependent_band_gap.value = [1.0, 2.0, 3.0, 4.0] * ureg.joule
        t_dependent_band_gap.normalize(EntryArchive(), logger)
        assert t_dependent_band_gap.type == 'direct'
        assert (
            np.isclose(t_dependent_band_gap.value.magnitude, [1.0, 2.0, 3.0, 4.0])
        ).all()
