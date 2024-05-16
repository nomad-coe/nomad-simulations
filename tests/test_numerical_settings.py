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
from typing import Optional, List

from nomad.units import ureg
from nomad.datamodel import EntryArchive

from nomad_simulations.numerical_settings import KSpace, KMesh, KLinePath

from . import logger
from .conftest import generate_k_space_simulation


class TestKSpace:
    """
    Test the `KSpace` class defined in `numerical_settings.py`.
    """

    @pytest.mark.parametrize(
        'system_type, is_representative, reciprocal_lattice_vectors, result',
        [
            ('bulk', False, None, None),
            ('atom', True, None, None),
            ('bulk', True, None, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            (
                'bulk',
                True,
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            ),
        ],
    )
    def test_normalize(
        self,
        system_type: Optional[str],
        is_representative: bool,
        reciprocal_lattice_vectors: Optional[List[List[float]]],
        result: List[List[float]],
    ):
        """
        Test the `normalize` method. This also test the `resolve_reciprocal_lattice_vectors` method.
        """
        simulation = generate_k_space_simulation(
            system_type=system_type,
            is_representative=is_representative,
            reciprocal_lattice_vectors=reciprocal_lattice_vectors,
        )
        k_space = simulation.model_method[0].numerical_settings[0]
        assert k_space.name == 'KSpace'
        k_space.normalize(EntryArchive(), logger)
        if k_space.reciprocal_lattice_vectors is not None:
            value = k_space.reciprocal_lattice_vectors.to('1/angstrom').magnitude / (
                2 * np.pi
            )
            assert np.allclose(value, result)
        else:
            assert k_space.reciprocal_lattice_vectors == result


class TestKLinePath:
    """
    Test the `KLinePath` class defined in `numerical_settings.py`.
    """

    def test_get_high_symmetry_points_norm(self, k_line_path: KLinePath):
        """
        Test the `get_high_symmetry_points_norm` method.
        """
        rlv = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * ureg('1/meter')
        high_symmetry_points_norms = k_line_path.get_high_symmetry_points_norm(
            reciprocal_lattice_vectors=rlv
        )
        hs_points = {
            'Gamma1': 0,
            'X': 0.5,
            'Y': 0.5 + 1 / np.sqrt(2),
            'Gamma2': 1 + 1 / np.sqrt(2),
        }
        for key, val in hs_points.items():
            assert np.isclose(high_symmetry_points_norms[key].magnitude, val)

    def test_resolve_points(self, k_line_path: KLinePath):
        """
        Test the `resolve_points` method.
        """
        rlv = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * ureg('1/meter')
        hs_points = {
            'Gamma1': 0,
            'X': 0.5,
            'Y': 0.5 + 1 / np.sqrt(2),
            'Gamma2': 1 + 1 / np.sqrt(2),
        }
        # Define paths
        gamma_x = np.linspace(hs_points['Gamma1'], hs_points['X'], num=5)
        x_y = np.linspace(hs_points['X'], hs_points['Y'], num=5)
        y_gamma = np.linspace(hs_points['Y'], hs_points['Gamma2'], num=5)
        points_norm = np.concatenate((gamma_x, x_y, y_gamma))
        k_line_path.resolve_points(
            points_norm=points_norm, reciprocal_lattice_vectors=rlv, logger=logger
        )
        assert len(points_norm) == len(k_line_path.points)
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # 'Gamma'
                [0.125, 0.0, 0.0],
                [0.25, 0.0, 0.0],
                [0.375, 0.0, 0.0],
                [0.5, 0.0, 0.0],  # 'X'
                [0.4, 0.1, 0.0],
                [0.3, 0.2, 0.0],
                [0.2, 0.3, 0.0],
                [0.1, 0.4, 0.0],
                [0.0, 0.5, 0.0],  # 'Y'
                [0.0, 0.4, 0.0],
                [0.0, 0.3, 0.0],
                [0.0, 0.2, 0.0],
                [0.0, 0.1, 0.0],
                [0.0, 0.0, 0.0],  # 'Gamma'
            ]
        )
        assert np.allclose(k_line_path.points, points)
