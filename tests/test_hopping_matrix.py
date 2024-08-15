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

from typing import Optional

import pytest

from nomad_simulations.schema_packages.properties import (
    CrystalFieldSplitting,
    HoppingMatrix,
)


class TestHoppingMatrix:
    """
    Test the `HoppingMatrix` class defined in `properties/hopping_matrix.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    @pytest.mark.parametrize(
        'n_orbitals, rank',
        [
            (None, None),
            (3, [3, 3]),
        ],
    )
    def test_default_quantities(self, n_orbitals: Optional[int], rank: Optional[list]):
        """
        Test the default quantities assigned when creating an instance of the `HoppingMatrix` class.
        """
        if n_orbitals is None:
            with pytest.raises(ValueError) as exc_info:
                hopping_matrix = HoppingMatrix(n_orbitals=n_orbitals)
            assert (
                str(exc_info.value)
                == '`n_orbitals` is not defined during initialization of the class.'
            )
        else:
            hopping_matrix = HoppingMatrix(n_orbitals=n_orbitals)
            assert hopping_matrix.iri == 'http://fairmat-nfdi.eu/taxonomy/HoppingMatrix'
            assert hopping_matrix.name == 'HoppingMatrix'
            assert hopping_matrix.rank == rank


class TestCrystalFieldSplitting:
    """
    Test the `CrystalFieldSplitting` class defined in `properties/hopping_matrix.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    @pytest.mark.parametrize(
        'n_orbitals, rank',
        [
            (None, None),
            (3, [3]),
        ],
    )
    def test_default_quantities(self, n_orbitals: Optional[int], rank: Optional[list]):
        """
        Test the default quantities assigned when creating an instance of the `CrystalFieldSplitting` class.
        """
        if n_orbitals is None:
            with pytest.raises(ValueError) as exc_info:
                crystal_field = CrystalFieldSplitting(n_orbitals=n_orbitals)
            assert (
                str(exc_info.value)
                == '`n_orbitals` is not defined during initialization of the class.'
            )
        else:
            crystal_field = CrystalFieldSplitting(n_orbitals=n_orbitals)
            assert (
                crystal_field.iri
                == 'http://fairmat-nfdi.eu/taxonomy/CrystalFieldSplitting'
            )
            assert crystal_field.name == 'CrystalFieldSplitting'
            assert crystal_field.rank == rank
