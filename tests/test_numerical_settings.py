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

import numpy as np
import pytest
from nomad.datamodel import EntryArchive
from nomad.units import ureg

from nomad_simulations.schema_packages.numerical_settings import (
    KLinePath,
    KMesh,
    KSpaceFunctionalities,
)

from . import logger
from .conftest import generate_k_line_path, generate_k_space_simulation


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
        reciprocal_lattice_vectors: Optional[list[list[float]]],
        result: list[list[float]],
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


class TestKSpaceFunctionalities:
    """
    Test the `KSpaceFunctionalities` class defined in `numerical_settings.py`.
    """

    @pytest.mark.parametrize(
        'reciprocal_lattice_vectors, check_grid, grid, result',
        [
            (None, None, None, False),
            ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], False, None, True),
            ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], True, None, False),
            ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], True, [6, 6, 6, 4], False),
            ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], True, [6, 6, 6], True),
        ],
    )
    def test_validate_reciprocal_lattice_vectors(
        self,
        reciprocal_lattice_vectors: Optional[list[list[float]]],
        check_grid: bool,
        grid: Optional[list[int]],
        result: bool,
    ):
        """
        Test the `validate_reciprocal_lattice_vectors` method.
        """
        check = KSpaceFunctionalities().validate_reciprocal_lattice_vectors(
            reciprocal_lattice_vectors=reciprocal_lattice_vectors,
            logger=logger,
            check_grid=check_grid,
            grid=grid,
        )
        assert check == result

    def test_resolve_high_symmetry_points(self):
        """
        Test the `resolve_high_symmetry_points` method. Only testing the valid situation in which the `ModelSystem` normalization worked.
        """
        # `ModelSystem.normalize()` need to extract `bulk` as a type.
        simulation = generate_k_space_simulation(
            pbc=[True, True, True],
        )
        model_systems = simulation.model_system
        # normalize to extract symmetry
        simulation.model_system[0].normalize(EntryArchive(), logger)

        # Testing the functionality method
        high_symmetry_points = KSpaceFunctionalities().resolve_high_symmetry_points(
            model_systems=model_systems, logger=logger
        )
        assert len(high_symmetry_points) == 4
        assert high_symmetry_points == {
            'Gamma': [0, 0, 0],
            'M': [0.5, 0.5, 0],
            'R': [0.5, 0.5, 0.5],
            'X': [0, 0.5, 0],
        }


class TestKMesh:
    """
    Test the `KMesh` class defined in `numerical_settings.py`.
    """

    @pytest.mark.parametrize(
        'center, grid, result_points, result_offset',
        [
            # No `center` and `grid`
            (None, None, None, None),
            # No `grid`
            ('Gamma-centered', None, None, None),
            ('Monkhorst-Pack', None, None, None),
            # `center` is `'Gamma-centered'`
            (
                'Gamma-centered',
                [2, 2, 2],
                [[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]],  # ! this result is weird @ndaelman-hu
                [0.0, 0.0, 0.0],
            ),
            # `center` is `'Monkhorst-Pack'`
            (
                'Monkhorst-Pack',
                [2, 2, 2],
                [
                    [-0.25, -0.25, -0.25],
                    [-0.25, -0.25, 0.25],
                    [-0.25, 0.25, -0.25],
                    [-0.25, 0.25, 0.25],
                    [0.25, -0.25, -0.25],
                    [0.25, -0.25, 0.25],
                    [0.25, 0.25, -0.25],
                    [0.25, 0.25, 0.25],
                ],
                [0.0, 0.0, 0.0],
            ),
            # Invalid `grid`
            ('Monkhorst-Pack', [-2, 2, 2], None, None),
        ],
    )
    def test_resolve_points_and_offset(
        self, center, grid, result_points, result_offset
    ):
        """
        Test the `resolve_points_and_offset` method.
        """
        k_mesh = KMesh(center=center)
        if grid is not None:
            k_mesh.grid = grid
        points, offset = k_mesh.resolve_points_and_offset(logger=logger)
        if points is not None:
            assert np.allclose(points, result_points)
        else:
            assert points == result_points
        if offset is not None:
            assert np.allclose(offset, result_offset)
        else:
            assert offset == result_offset

    @pytest.mark.parametrize(
        'system_type, is_representative, grid, reciprocal_lattice_vectors, result_get_k_line_density, result_k_line_density',
        [
            # No `grid` and `reciprocal_lattice_vectors`
            ('bulk', False, None, None, None, None),
            # No `reciprocal_lattice_vectors`
            ('bulk', False, [6, 6, 6], None, None, None),
            # No `grid`
            ('bulk', False, None, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], None, None),
            # `is_representative` set to False
            (
                'bulk',
                False,
                [6, 6, 6],
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                0.954929658,
                None,
            ),
            # `system_type` is not 'bulk'
            (
                'atom',
                True,
                [6, 6, 6],
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                0.954929658,
                None,
            ),
            # All parameters are set
            (
                'bulk',
                True,
                [6, 6, 6],
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                0.954929658,
                0.954929658,
            ),
        ],
    )
    def test_resolve_k_line_density(
        self,
        system_type: Optional[str],
        is_representative: bool,
        grid: Optional[list[int]],
        reciprocal_lattice_vectors: Optional[list[list[float]]],
        result_get_k_line_density: Optional[float],
        result_k_line_density: Optional[float],
    ):
        """
        Test the `resolve_k_line_density` and `get_k_line_density` methods
        """
        simulation = generate_k_space_simulation(
            system_type=system_type,
            is_representative=is_representative,
            reciprocal_lattice_vectors=reciprocal_lattice_vectors,
            grid=grid,
        )
        k_space = simulation.model_method[0].numerical_settings[0]
        reciprocal_lattice_vectors = k_space.reciprocal_lattice_vectors
        k_mesh = k_space.k_mesh[0]
        model_systems = simulation.model_system
        # Applying method `get_k_line_density`
        get_k_line_density_value = k_mesh.get_k_line_density(
            reciprocal_lattice_vectors=reciprocal_lattice_vectors, logger=logger
        )
        if get_k_line_density_value is not None:
            assert np.isclose(
                get_k_line_density_value.to('angstrom').magnitude,
                result_get_k_line_density,
            )
        else:
            assert get_k_line_density_value == result_get_k_line_density
        # Applying method `resolve_k_line_density`
        k_line_density = k_mesh.resolve_k_line_density(
            model_systems=model_systems,
            reciprocal_lattice_vectors=reciprocal_lattice_vectors,
            logger=logger,
        )
        if k_line_density is not None:
            assert np.isclose(
                k_line_density.to('angstrom').magnitude, result_k_line_density
            )
        else:
            assert k_line_density == result_k_line_density


class TestKLinePath:
    """
    Test the `KLinePath` class defined in `numerical_settings.py`.
    """

    @pytest.mark.parametrize(
        'high_symmetry_path_names, high_symmetry_path_values, result',
        [
            (None, None, False),
            (['Gamma', 'X', 'Y'], None, False),
            ([], [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0]], False),
            (['Gamma', 'X', 'Y'], [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0]], True),
        ],
    )
    def test_validate_high_symmetry_path(
        self,
        high_symmetry_path_names: list[str],
        high_symmetry_path_values: list[list[float]],
        result: bool,
    ):
        """
        Test the `validate_high_symmetry_path` method.
        """
        k_line_path = generate_k_line_path(
            high_symmetry_path_names=high_symmetry_path_names,
            high_symmetry_path_values=high_symmetry_path_values,
        )
        assert k_line_path.validate_high_symmetry_path(logger=logger) == result

    @pytest.mark.parametrize(
        'reciprocal_lattice_vectors, high_symmetry_path_names, result',
        [
            (None, None, []),
            ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], None, []),
            (
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                ['Gamma', 'X', 'R'],
                [[0, 0, 0], [0, 0.5, 0], [0.5, 0.5, 0.5]],
            ),
        ],
    )
    def test_resolve_high_symmetry_path_values(
        self,
        reciprocal_lattice_vectors: Optional[list[list[float]]],
        high_symmetry_path_names: list[str],
        result: list[float],
    ):
        """
        Test the `resolve_high_symmetry_path_values` method. Only testing the valid situation in which the `ModelSystem` normalization worked.
        """
        # `ModelSystem.normalize()` need to extract `bulk` as a type.
        simulation = generate_k_space_simulation(
            pbc=[True, True, True],
            reciprocal_lattice_vectors=reciprocal_lattice_vectors,
            high_symmetry_path_names=high_symmetry_path_names,
            high_symmetry_path_values=None,
        )
        model_system = simulation.model_system[0]
        model_system.normalize(EntryArchive(), logger)  # normalize to extract symmetry

        # `KLinePath` can be understood as a `KMeshBase` section
        k_line_path = simulation.model_method[0].numerical_settings[0].k_line_path
        high_symmetry_points_values = k_line_path.resolve_high_symmetry_path_values(
            simulation.model_system, reciprocal_lattice_vectors, logger
        )
        assert high_symmetry_points_values == result

    def test_get_high_symmetry_path_norm(self, k_line_path: KLinePath):
        """
        Test the `get_high_symmetry_path_norm` method.
        """
        rlv = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * ureg('1/meter')
        high_symmetry_path_norms = k_line_path.get_high_symmetry_path_norms(
            reciprocal_lattice_vectors=rlv, logger=logger
        )
        hs_points = [0, 0.5, 0.5 + 1 / np.sqrt(2), 1 + 1 / np.sqrt(2)]
        for i, val in enumerate(hs_points):
            assert np.isclose(high_symmetry_path_norms[i].magnitude, val)

    def test_resolve_points(self, k_line_path: KLinePath):
        """
        Test the `resolve_points` method.
        """
        rlv = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * ureg('1/meter')
        hs_points = [0, 0.5, 0.5 + 1 / np.sqrt(2), 1 + 1 / np.sqrt(2)]
        # Define paths
        gamma_x = np.linspace(hs_points[0], hs_points[1], num=5)
        x_y = np.linspace(hs_points[1], hs_points[2], num=5)
        y_gamma = np.linspace(hs_points[2], hs_points[3], num=5)
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
