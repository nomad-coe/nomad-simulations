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
from typing import Optional, Tuple

from nomad_simulations.properties import ElectronicEigenvalues

from . import logger
from .conftest import generate_electronic_eigenvalues


class TestElectronicEigenvalues:
    """
    Test the `ElectronicEigenvalues` class defined in `properties/band_structure.py`.
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
                electronic_eigenvalues = ElectronicEigenvalues(n_bands=n_bands)
            assert (
                str(exc_info.value)
                == f'`n_bands` is not defined during initialization of the class.'
            )
        else:
            electronic_eigenvalues = ElectronicEigenvalues(n_bands=n_bands)
            assert (
                electronic_eigenvalues.iri
                == 'http://fairmat-nfdi.eu/taxonomy/ElectronicEigenvalues'
            )
            assert electronic_eigenvalues.name == 'ElectronicEigenvalues'
            assert electronic_eigenvalues.rank == rank

    @pytest.mark.parametrize(
        'occupation, result_check',
        [
            (None, False),
            ([], False),
            ([[2, 2], [0, 0]], False),  # `value` and `occupation` must have same shape
            (
                [[0, 2], [0, 1], [0, 2], [0, 2], [0, 1.5], [0, 1.5], [0, 1], [0, 2]],
                True,
            ),
        ],
    )
    def test_check_occupation_shape(
        self, occupation: Optional[list], result_check: bool
    ):
        """
        Test the `_check_occupation_shape` private method.
        """
        electronic_eigenvalues = generate_electronic_eigenvalues(
            value=[[3, -2], [3, 1], [4, -2], [5, -1], [4, 0], [2, 0], [2, 1], [4, -3]],
            occupation=occupation,
        )
        assert electronic_eigenvalues._check_occupation_shape(logger) == result_check

    @pytest.mark.parametrize(
        'occupation, value, result',
        [
            (
                None,
                [[3, -2], [3, 1], [4, -2], [5, -1], [4, 0], [2, 0], [2, 1], [4, -3]],
                (None, None),
            ),
            (
                [],
                [[3, -2], [3, 1], [4, -2], [5, -1], [4, 0], [2, 0], [2, 1], [4, -3]],
                (None, None),
            ),
            (
                [[2, 2], [0, 0]],
                [[3, -2], [3, 1], [4, -2], [5, -1], [4, 0], [2, 0], [2, 1], [4, -3]],
                (None, None),
            ),  # `value` and `occupation` must have same shape
            (
                [[0, 2], [0, 1], [0, 2], [0, 2], [0, 1.5], [0, 1.5], [0, 1], [0, 2]],
                None,
                (None, None),
            ),
            (
                [[0, 2], [0, 1], [0, 2], [0, 2], [0, 1.5], [0, 1.5], [0, 1], [0, 2]],
                [[3, -2], [3, 1], [4, -2], [5, -1], [4, 0], [2, 0], [2, 1], [4, -3]],
                (
                    np.array(
                        [
                            -3,
                            -2,
                            -2,
                            -1,
                            0,
                            0,
                            1,
                            1,
                            2,
                            2,
                            3,
                            3,
                            4,
                            4,
                            4,
                            5,
                        ]
                    ),
                    np.array(
                        [
                            2.0,
                            2.0,
                            2.0,
                            2.0,
                            1.5,
                            1.5,
                            1.0,
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ]
                    ),
                ),
            ),
        ],
    )
    def test_order_eigenvalues(
        self,
        occupation: Optional[list],
        value: Optional[list],
        result: Tuple[Optional[list], Optional[list]],
    ):
        """
        Test the `order_eigenvalues` method.
        """
        electronic_eigenvalues = generate_electronic_eigenvalues(
            value=value,
            occupation=occupation,
        )
        sorted_value, sorted_occupation = electronic_eigenvalues.order_eigenvalues(
            logger
        )
        if sorted_value is not None and sorted_occupation is not None:
            assert electronic_eigenvalues.m_cache['sorted_eigenvalues']
            assert (sorted_value.magnitude == result[0]).all()
            assert (sorted_occupation == result[1]).all()
        else:
            assert (sorted_value, sorted_occupation) == result

    @pytest.mark.parametrize(
        'occupation, value, highest_occupied, lowest_unoccupied, result',
        [
            # Not possible to resolve `highest_occupied` and `lowest_unoccupied`
            (
                None,
                [[3, -2], [3, 1], [4, -2], [5, -1], [4, 0], [2, 0], [2, 1], [4, -3]],
                None,
                None,
                (None, None),
            ),
            (
                [],
                [[3, -2], [3, 1], [4, -2], [5, -1], [4, 0], [2, 0], [2, 1], [4, -3]],
                None,
                None,
                (None, None),
            ),
            (
                [[2, 2], [0, 0]],
                [[3, -2], [3, 1], [4, -2], [5, -1], [4, 0], [2, 0], [2, 1], [4, -3]],
                None,
                None,
                (None, None),
            ),  # `value` and `occupation` must have same shape
            (
                [[0, 2], [0, 1], [0, 2], [0, 2], [0, 1.5], [0, 1.5], [0, 1], [0, 2]],
                None,
                None,
                None,
                (None, None),
            ),
            # `highest_occupied` and `lowest_unoccupied` are passed to the class
            (
                None,
                [[3, -2], [3, 1], [4, -2], [5, -1], [4, 0], [2, 0], [2, 1], [4, -3]],
                1.0,
                2.0,
                (1.0, 2.0),
            ),
            (
                [],
                [[3, -2], [3, 1], [4, -2], [5, -1], [4, 0], [2, 0], [2, 1], [4, -3]],
                1.0,
                2.0,
                (1.0, 2.0),
            ),
            (
                [[2, 2], [0, 0]],
                [[3, -2], [3, 1], [4, -2], [5, -1], [4, 0], [2, 0], [2, 1], [4, -3]],
                1.0,
                2.0,
                (1.0, 2.0),
            ),
            (
                [[0, 2], [0, 1], [0, 2], [0, 2], [0, 1.5], [0, 1.5], [0, 1], [0, 2]],
                None,
                1.0,
                2.0,
                (1.0, 2.0),
            ),
            # Resolving `highest_occupied` and `lowest_unoccupied` from `value` and `occupation`
            (
                [[0, 2], [0, 1], [0, 2], [0, 2], [0, 1.5], [0, 1.5], [0, 1], [0, 2]],
                [[3, -2], [3, 1], [4, -2], [5, -1], [4, 0], [2, 0], [2, 1], [4, -3]],
                None,
                None,
                (1.0, 2.0),
            ),
            # Overwritting stored `highest_occupied` and `lowest_unoccupied` from `value` and `occupation`
            (
                [[0, 2], [0, 1], [0, 2], [0, 2], [0, 1.5], [0, 1.5], [0, 1], [0, 2]],
                [[3, -2], [3, 1], [4, -2], [5, -1], [4, 0], [2, 0], [2, 1], [4, -3]],
                -3.0,
                4.0,
                (1.0, 2.0),
            ),
        ],
    )
    def test_homo_lumo_eigenvalues(
        self,
        occupation: Optional[list],
        value: Optional[list],
        highest_occupied: Optional[float],
        lowest_unoccupied: Optional[float],
        result: Tuple[Optional[float], Optional[float]],
    ):
        """
        Test the `resolve_homo_lumo_eigenvalues` method.
        """
        electronic_eigenvalues = generate_electronic_eigenvalues(
            value=value,
            occupation=occupation,
            highest_occupied=highest_occupied,
            lowest_unoccupied=lowest_unoccupied,
        )
        homo, lumo = electronic_eigenvalues.resolve_homo_lumo_eigenvalues(logger)
        if homo is not None and lumo is not None:
            assert (homo.magnitude, lumo.magnitude) == result
        else:
            assert (homo, lumo) == result

    @pytest.mark.parametrize(
        'occupation, value, highest_occupied, lowest_unoccupied, band_gap_result',
        [
            # Not possible to resolve `highest_occupied` and `lowest_unoccupied`
            (
                None,
                [[3, -2], [3, 1], [4, -2], [5, -1], [4, 0], [2, 0], [2, 1], [4, -3]],
                None,
                None,
                None,
            ),
            (
                [],
                [[3, -2], [3, 1], [4, -2], [5, -1], [4, 0], [2, 0], [2, 1], [4, -3]],
                None,
                None,
                None,
            ),
            (
                [[2, 2], [0, 0]],
                [[3, -2], [3, 1], [4, -2], [5, -1], [4, 0], [2, 0], [2, 1], [4, -3]],
                None,
                None,
                None,
            ),  # `value` and `occupation` must have same shape
            (
                [[0, 2], [0, 1], [0, 2], [0, 2], [0, 1.5], [0, 1.5], [0, 1], [0, 2]],
                None,
                None,
                None,
                None,
            ),
            # `highest_occupied` and `lowest_unoccupied` are passed to the class
            (
                None,
                [[3, -2], [3, 1], [4, -2], [5, -1], [4, 0], [2, 0], [2, 1], [4, -3]],
                1.0,
                2.0,
                1.0,
            ),
            (
                [],
                [[3, -2], [3, 1], [4, -2], [5, -1], [4, 0], [2, 0], [2, 1], [4, -3]],
                1.0,
                2.0,
                1.0,
            ),
            (
                [[2, 2], [0, 0]],
                [[3, -2], [3, 1], [4, -2], [5, -1], [4, 0], [2, 0], [2, 1], [4, -3]],
                1.0,
                2.0,
                1.0,
            ),
            (
                [[0, 2], [0, 1], [0, 2], [0, 2], [0, 1.5], [0, 1.5], [0, 1], [0, 2]],
                None,
                1.0,
                2.0,
                1.0,
            ),
            # If (lumo - homo) is negative, band_gap_result is 0
            (
                [[0, 2], [0, 1], [0, 2], [0, 2], [0, 1.5], [0, 1.5], [0, 1], [0, 2]],
                None,
                3.0,
                2.0,
                0.0,
            ),
            # Resolving `highest_occupied` and `lowest_unoccupied` from `value` and `occupation`
            (
                [[0, 2], [0, 1], [0, 2], [0, 2], [0, 1.5], [0, 1.5], [0, 1], [0, 2]],
                [[3, -2], [3, 1], [4, -2], [5, -1], [4, 0], [2, 0], [2, 1], [4, -3]],
                None,
                None,
                1.0,
            ),
            # Overwritting stored `highest_occupied` and `lowest_unoccupied` from `value` and `occupation`
            (
                [[0, 2], [0, 1], [0, 2], [0, 2], [0, 1.5], [0, 1.5], [0, 1], [0, 2]],
                [[3, -2], [3, 1], [4, -2], [5, -1], [4, 0], [2, 0], [2, 1], [4, -3]],
                -3.0,
                4.0,
                1.0,
            ),
        ],
    )
    def test_extract_band_gap(
        self,
        occupation: Optional[list],
        value: Optional[list],
        highest_occupied: Optional[float],
        lowest_unoccupied: Optional[float],
        band_gap_result: Optional[float],
    ):
        """
        Test the `extract_band_gap` method.
        """
        electronic_eigenvalues = generate_electronic_eigenvalues(
            value=value,
            occupation=occupation,
            highest_occupied=highest_occupied,
            lowest_unoccupied=lowest_unoccupied,
        )
        band_gap = electronic_eigenvalues.extract_band_gap(logger)
        if band_gap is not None:
            assert np.isclose(band_gap.value.magnitude, band_gap_result)
        else:
            assert band_gap == band_gap_result

    def test_extract_fermi_surface(self):
        """
        Test the `extract_band_gap` method.
        """
        # ! add test when `FermiSurface` is implemented
        pass
