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

import pytest
from nomad.datamodel import EntryArchive

from nomad_simulations.schema_packages.properties import (
    QuasiparticleWeight,
)
from nomad_simulations.schema_packages.properties.greens_function import (
    BaseGreensFunction,
)
from nomad_simulations.schema_packages.variables import (
    Frequency,
    ImaginaryTime,
    KMesh,
    MatsubaraFrequency,
    Time,
    WignerSeitz,
)

from . import logger


class TestBaseGreensFunction:
    """
    Test the `BaseGreensFunction` class defined in `properties/greens_function.py`.aa
    """

    @pytest.mark.parametrize(
        'variables, result',
        [
            ([], None),
            ([WignerSeitz()], 'r'),
            ([KMesh()], 'k'),
            ([Time()], 't'),
            ([ImaginaryTime()], 'it'),
            ([Frequency()], 'w'),
            ([MatsubaraFrequency()], 'iw'),
            ([WignerSeitz(), Time()], 'rt'),
            ([WignerSeitz(), ImaginaryTime()], 'rit'),
            ([WignerSeitz(), Frequency()], 'rw'),
            ([WignerSeitz(), MatsubaraFrequency()], 'riw'),
            ([KMesh(), Time()], 'kt'),
            ([KMesh(), ImaginaryTime()], 'kit'),
            ([KMesh(), Frequency()], 'kw'),
            ([KMesh(), MatsubaraFrequency()], 'kiw'),
        ],
    )
    def test_resolve_space_id(
        self,
        variables: list[
            Union[
                WignerSeitz, KMesh, Time, ImaginaryTime, Frequency, MatsubaraFrequency
            ]
        ],
        result: str,
    ):
        """
        Test the `resolve_space_id` method of the `BaseGreensFunction` class.
        """
        gfs = BaseGreensFunction(n_atoms=1, n_correlated_orbitals=1)
        gfs.variables = variables
        assert gfs.resolve_space_id() == result

    @pytest.mark.parametrize(
        'space_id, variables, result',
        [
            ('', [], None),  # empty `space_id`
            ('rt', [], None),  # `space_id` set by parser
            ('', [WignerSeitz()], 'r'),  # resolving `space_id`
            ('rt', [WignerSeitz()], 'r'),  # normalize overwrites `space_id`
            ('', [KMesh()], 'k'),
            ('', [Time()], 't'),
            ('', [ImaginaryTime()], 'it'),
            ('', [Frequency()], 'w'),
            ('', [MatsubaraFrequency()], 'iw'),
            ('', [WignerSeitz(), Time()], 'rt'),
            ('', [WignerSeitz(), ImaginaryTime()], 'rit'),
            ('', [WignerSeitz(), Frequency()], 'rw'),
            ('', [WignerSeitz(), MatsubaraFrequency()], 'riw'),
            ('', [KMesh(), Time()], 'kt'),
            ('', [KMesh(), ImaginaryTime()], 'kit'),
            ('', [KMesh(), Frequency()], 'kw'),
            ('', [KMesh(), MatsubaraFrequency()], 'kiw'),
        ],
    )
    def test_normalize(
        self,
        space_id: str,
        variables: list[
            Union[
                WignerSeitz, KMesh, Time, ImaginaryTime, Frequency, MatsubaraFrequency
            ]
        ],
        result: Optional[str],
    ):
        """
        Test the `normalize` method of the `BaseGreensFunction` class.
        """
        gfs = BaseGreensFunction(n_atoms=1, n_correlated_orbitals=1)
        gfs.variables = variables
        gfs.space_id = space_id if space_id else None
        gfs.normalize(archive=EntryArchive(), logger=logger)
        assert gfs.space_id == result


class TestQuasiparticleWeight:
    """
    Test the `QuasiparticleWeight` class defined in `properties/greens_function.py`.
    """

    @pytest.mark.parametrize(
        'value, result',
        [
            ([[1, 0.5, -2]], False),
            ([[1, 0.5, 8]], False),
            ([[1, 0.5, 0.8]], True),
        ],
    )
    def test_is_valid_quasiparticle_weight(self, value: list[float], result: bool):
        """
        Test the `is_valid_quasiparticle_weight` method of the `QuasiparticleWeight` class.
        """
        quasiparticle_weight = QuasiparticleWeight(n_atoms=1, n_correlated_orbitals=3)
        quasiparticle_weight.value = value
        assert quasiparticle_weight.is_valid_quasiparticle_weight() == result

    @pytest.mark.parametrize(
        'value, result',
        [
            ([[1, 0.9, 0.8]], 'non-correlated metal'),
            ([[0.2, 0.3, 0.1]], 'strongly-correlated metal'),
            ([[0, 0.3, 0.1]], 'OSMI'),
            ([[0, 0, 0]], 'Mott insulator'),
            ([[1.0, 0.8, 0.2]], None),
        ],
    )
    def test_resolve_system_correlation_strengths(
        self, value: list[float], result: Optional[str]
    ):
        """
        Test the `resolve_system_correlation_strengths` method of the `QuasiparticleWeight` class.
        """
        quasiparticle_weight = QuasiparticleWeight(n_atoms=1, n_correlated_orbitals=3)
        quasiparticle_weight.value = value
        assert quasiparticle_weight.resolve_system_correlation_strengths() == result

    @pytest.mark.parametrize(
        'value, result',
        [
            ([[1, 0.5, -2]], None),
            ([[1, 0.5, 8]], None),
            ([[1, 0.9, 0.8]], 'non-correlated metal'),
            ([[0.2, 0.3, 0.1]], 'strongly-correlated metal'),
            ([[0, 0.3, 0.1]], 'OSMI'),
            ([[0, 0, 0]], 'Mott insulator'),
            ([[1.0, 0.8, 0.2]], None),
        ],
    )
    def test_normalize(self, value: list[float], result: Optional[str]):
        """
        Test the `normalize` method of the `QuasiparticleWeight` class.
        """
        quasiparticle_weight = QuasiparticleWeight(n_atoms=1, n_correlated_orbitals=3)
        quasiparticle_weight.value = value
        quasiparticle_weight.normalize(archive=EntryArchive(), logger=logger)
        assert quasiparticle_weight.system_correlation_strengths == result
