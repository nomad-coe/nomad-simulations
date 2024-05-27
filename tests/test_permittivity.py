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
from typing import List, Optional

from nomad.units import ureg
from nomad.datamodel import EntryArchive

from . import logger

from nomad_simulations import Simulation
from nomad_simulations.model_system import ModelSystem, AtomicCell
from nomad_simulations.atoms_state import AtomsState
from nomad_simulations.outputs import Outputs
from nomad_simulations.properties import Permittivity
from nomad_simulations.variables import Variables, KMesh, Frequency


class TestPermittivity:
    """
    Test the `Permittivity` class defined in `properties/permittivity.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `Permittivity` class.
        """
        permittivity = Permittivity()
        assert permittivity.iri == 'http://fairmat-nfdi.eu/taxonomy/Permittivity'
        assert permittivity.name == 'Permittivity'
        assert permittivity.rank == [3, 3]

    @pytest.mark.parametrize(
        'variables, result',
        [
            (None, 'static'),
            ([], 'static'),
            ([KMesh()], 'static'),
            ([KMesh(), Frequency()], 'dynamic'),
        ],
    )
    def test_resolve_type(self, variables: Optional[List[Variables]], result: str):
        """
        Test the `resolve_type` method.
        """
        permittivity = Permittivity()
        if variables is not None:
            permittivity.variables = [var for var in variables]
        assert permittivity.resolve_type() == result

    @pytest.mark.parametrize(
        'variables, value, result',
        [
            # Empty case
            (None, None, None),
            # No `variables`
            ([], np.eye(3) * (1 + 1j), None),
            # If `variables` contain `KMesh`, we cannot extract absorption spectra
            (
                [KMesh(points=[[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])],
                np.array([np.eye(3) * k_point * (1 + 1j) for k_point in range(1, 5)]),
                None,
            ),
            # Even if `variables` contain `Frequency`, we cannot extract absorption spectra if `value` depends on `KMesh`
            (
                [
                    Frequency(points=[0, 1, 2, 3, 4]),
                    KMesh(points=[[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]),
                ],
                np.array(
                    [
                        [
                            np.eye(3) * k_point * (1 + 1j)
                            + np.eye(3) * freq_point * 0.5j
                            for k_point in range(1, 5)
                        ]
                        for freq_point in range(5)
                    ]
                ),
                None,
            ),
            # Valid case: `value` does not depend on `KMesh` and we can extract absorption spectra
            (
                [
                    Frequency(points=[0, 1, 2, 3, 4]),
                ],
                np.array([np.eye(3) * freq_point * 0.5j for freq_point in range(5)]),
                [0.0, 0.5, 1.0, 1.5, 2.0],
            ),
        ],
    )
    def test_extract_absorption_spectra(
        self,
        variables: Optional[List[Variables]],
        value: Optional[np.ndarray],
        result: Optional[List[float]],
    ):
        """
        Test the `extract_absorption_spectra` method. The `result` in the last valid case corresponds to the imaginary part of
        the diagonal of the `Permittivity.value` for each frequency point.
        """
        permittivity = Permittivity()
        if variables is not None:
            permittivity.variables = [var for var in variables]
            permittivity.value = value
        absorption_spectra = permittivity.extract_absorption_spectra(logger)
        if absorption_spectra is not None:
            assert len(absorption_spectra) == 3
            spectrum = absorption_spectra[1]
            assert spectrum.rank == []
            assert spectrum.axis == 'yy'
            assert len(spectrum.value) == len(variables[0].points)
            assert np.allclose(spectrum.value, result)
        else:
            assert absorption_spectra == result
