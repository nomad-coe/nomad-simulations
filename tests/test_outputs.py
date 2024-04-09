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

from . import logger
from .conftest import get_scf_electronic_band_gap_template

from nomad.units import ureg
from nomad.metainfo import Quantity
from nomad_simulations.physical_property import PhysicalProperty
from nomad_simulations.numerical_settings import SelfConsistency
from nomad_simulations.outputs import Outputs, SCFOutputs, ElectronicBandGap


class TotalEnergy(PhysicalProperty):
    """Physical property class defined for testing purposes."""

    shape = []
    variables = []
    value = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        The total energy of the system.
        """,
    )


class TestOutputs:
    """
    Test the `Outputs` class defined in `outputs.py`.
    """

    @pytest.mark.parametrize(
        'threshold_change, result',
        [(1e-3, True), (1e-5, False)],
    )
    def test_is_scf_converged(self, threshold_change: float, result: bool):
        """
        Test the  `resolve_is_scf_converged` method.
        """
        scf_outputs = get_scf_electronic_band_gap_template(
            threshold_change=threshold_change
        )
        is_scf_converged = scf_outputs.resolve_is_scf_converged(
            property_name='electronic_band_gap',
            i_property=0,
            phys_property=scf_outputs.electronic_band_gap[0],
            logger=logger,
        )
        assert is_scf_converged == result

    @pytest.mark.parametrize(
        'threshold_change, result',
        [(1e-3, True), (1e-5, False)],
    )
    def test_normalize(self, threshold_change: float, result: bool):
        """
        Test the `normalize` method.
        """
        scf_outputs = get_scf_electronic_band_gap_template(
            threshold_change=threshold_change
        )
        # Add a non-SCF calculated PhysicalProperty
        scf_outputs.custom_physical_property = [
            TotalEnergy(name='TotalEnergy', value=1 * ureg.joule)
        ]

        scf_outputs.normalize(None, logger)
        assert scf_outputs.electronic_band_gap[0].is_scf_converged == result
        assert scf_outputs.custom_physical_property[0].is_scf_converged is None
