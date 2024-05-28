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

from nomad.units import ureg
from nomad.datamodel import EntryArchive

from . import logger
from .conftest import generate_scf_electronic_band_gap_template

from nomad_simulations.outputs import Outputs, ElectronicBandGap


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
        scf_outputs = generate_scf_electronic_band_gap_template(
            threshold_change=threshold_change
        )
        is_scf_converged = scf_outputs.resolve_is_scf_converged(
            property_name='electronic_band_gaps',
            i_property=0,
            phys_property=scf_outputs.electronic_band_gaps[0],
            logger=logger,
        )
        assert is_scf_converged == result

    def test_extract_spin_polarized_properties(self):
        """
        Test the `extract_spin_polarized_property` method.
        """
        outputs = Outputs()

        # No spin-polarized band gap
        band_gap_non_spin_polarized = ElectronicBandGap(variables=[])
        band_gap_non_spin_polarized.value = 2.0 * ureg.joule
        outputs.electronic_band_gaps.append(band_gap_non_spin_polarized)
        band_gaps = outputs.extract_spin_polarized_property('electronic_band_gaps')
        assert band_gaps == []

        # Spin-polarized band gaps
        band_gap_spin_1 = ElectronicBandGap(variables=[], spin_channel=0)
        band_gap_spin_1.value = 1.0 * ureg.joule
        outputs.electronic_band_gaps.append(band_gap_spin_1)
        band_gap_spin_2 = ElectronicBandGap(variables=[], spin_channel=1)
        band_gap_spin_2.value = 1.5 * ureg.joule
        outputs.electronic_band_gaps.append(band_gap_spin_2)
        band_gaps = outputs.extract_spin_polarized_property('electronic_band_gaps')
        assert len(band_gaps) == 2
        assert band_gaps[0].value.magnitude == 1.0
        assert band_gaps[1].value.magnitude == 1.5

    @pytest.mark.parametrize(
        'threshold_change, result',
        [(1e-3, True), (1e-5, False)],
    )
    def test_normalize(self, threshold_change: float, result: bool):
        """
        Test the `normalize` method.
        """
        scf_outputs = generate_scf_electronic_band_gap_template(
            threshold_change=threshold_change
        )

        scf_outputs.normalize(EntryArchive(), logger)
        assert scf_outputs.electronic_band_gaps[0].is_scf_converged == result
