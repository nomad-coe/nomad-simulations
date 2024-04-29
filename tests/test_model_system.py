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
from .conftest import get_template_atomic_cell
from nomad_simulations.model_system import (
    Symmetry,
    ChemicalFormula,
    ModelSystem,
)
from nomad_simulations.atoms_state import AtomsState


class TestAtomicCell:
    """
    Test the `AtomicCell`, `Cell` and `GeometricSpace` classes defined in model_system.py
    """

    @pytest.mark.parametrize(
        'chemical_symbols, atomic_numbers, formula, lattice_vectors, positions, periodic_boundary_conditions',
        [
            (
                ['H', 'H', 'O'],
                [1, 1, 8],
                'H2O',
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]],
                [False, False, False],
            ),  # full atomic cell
            (
                [],
                [1, 1, 8],
                'H2O',
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]],
                [False, False, False],
            ),  # missing chemical_symbols
            (
                ['H', 'H', 'O'],
                [1, 1, 8],
                'H2O',
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [],
                [False, False, False],
            ),  # missing positions
            (
                ['H', 'H', 'O'],
                [1, 1, 8],
                'H2O',
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1], [2, 2, 2]],
                [False, False, False],
            ),  # chemical_symbols and positions with different lengths
            (
                ['H', 'H', 'O'],
                [1, 1, 8],
                'H2O',
                [],
                [[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]],
                [False, False, False],
            ),  # missing lattice_vectors
        ],
    )
    def test_generate_ase_atoms(
        self,
        chemical_symbols,
        atomic_numbers,
        formula,
        lattice_vectors,
        positions,
        periodic_boundary_conditions,
    ):
        """
        Test the creation of `ase.Atoms` from `AtomicCell`.
        """
        atomic_cell = get_template_atomic_cell(
            lattice_vectors,
            positions,
            periodic_boundary_conditions,
            chemical_symbols,
            atomic_numbers,
        )

        # Test `to_ase_atoms` function
        ase_atoms = atomic_cell.to_ase_atoms(logger)
        if not chemical_symbols or len(chemical_symbols) != len(positions):
            assert ase_atoms is None
        else:
            if lattice_vectors:
                assert (ase_atoms.cell == lattice_vectors).all()
            else:
                assert (ase_atoms.cell == [0, 0, 0]).all()
            assert (ase_atoms.positions == positions).all()
            assert (ase_atoms.pbc == periodic_boundary_conditions).all()
            assert (ase_atoms.symbols.numbers == atomic_numbers).all()
            assert ase_atoms.symbols.get_chemical_formula() == formula

    @pytest.mark.parametrize(
        'chemical_symbols, atomic_numbers, lattice_vectors, positions, vectors_results, angles_results, volume',
        [
            (
                ['H', 'H', 'O'],
                [1, 1, 8],
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]],
                [1.0, 1.0, 1.0],
                [90.0, 90.0, 90.0],
                1.0,
            ),  # full atomic cell
            (
                ['H', 'H', 'O'],
                [1, 1, 8],
                [[1.2, 2.3, 0], [1.2, -2.3, 0], [0, 0, 1]],
                [[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]],
                [2.59422435, 2.59422435, 1.0],
                [90.0, 90.0, 124.8943768],
                5.52,
            ),  # full atomic cell with different lattice_vectors
            (
                [],
                [1, 1, 8],
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]],
                [None, None, None],
                [None, None, None],
                None,
            ),  # missing chemical_symbols
            (
                ['H', 'H', 'O'],
                [1, 1, 8],
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [],
                [None, None, None],
                [None, None, None],
                None,
            ),  # missing positions
            (
                ['H', 'H', 'O'],
                [1, 1, 8],
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1], [2, 2, 2]],
                [None, None, None],
                [None, None, None],
                None,
            ),  # chemical_symbols and positions with different lengths
            (
                ['H', 'H', 'O'],
                [1, 1, 8],
                [],
                [[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]],
                [0.0, 0.0, 0.0],
                [90.0, 90.0, 90.0],
                0.0,
            ),  # missing lattice_vectors
        ],
    )
    def test_geometric_space(
        self,
        chemical_symbols,
        atomic_numbers,
        lattice_vectors,
        positions,
        vectors_results,
        angles_results,
        volume,
    ):
        """
        Test the `GeometricSpace` quantities normalization from `AtomicCell`.
        """
        pbc = [False, False, False]
        atomic_cell = get_template_atomic_cell(
            lattice_vectors,
            positions,
            pbc,
            chemical_symbols,
            atomic_numbers,
        )

        # Get `GeometricSpace` quantities via normalization of `AtomicCell`
        atomic_cell.normalize(None, logger)
        # Testing lengths of cell vectors
        for index, name in enumerate(
            ['length_vector_a', 'length_vector_b', 'length_vector_c']
        ):
            quantity = getattr(atomic_cell, name)
            if quantity is not None:
                assert np.isclose(
                    quantity.to('angstrom').magnitude,
                    vectors_results[index],
                )
            else:
                assert quantity == vectors_results[index]
        # Testing angles between cell vectors
        for index, name in enumerate(
            ['angle_vectors_b_c', 'angle_vectors_a_c', 'angle_vectors_a_b']
        ):
            quantity = getattr(atomic_cell, name)
            if quantity is not None:
                assert np.isclose(
                    quantity.to('degree').magnitude,
                    angles_results[index],
                )
            else:
                assert quantity == angles_results[index]
        # Testing volume
        if atomic_cell.volume is not None:
            assert np.isclose(atomic_cell.volume.to('angstrom^3').magnitude, volume)
        else:
            assert atomic_cell.volume == volume


class TestModelSystem:
    """
    Test the `ModelSystem`, `Symmetry` and `ChemicalFormula` classes defined in model_system.py
    """

    def test_empty_chemical_formula(self):
        """
        Test the empty `ChemicalFormula` normalization if a sibling `AtomicCell` is not provided.
        """
        chemical_formula = ChemicalFormula()
        chemical_formula.normalize(None, logger)
        for name in ['descriptive', 'reduced', 'iupac', 'hill', 'anonymous']:
            assert getattr(chemical_formula, name) is None

    @pytest.mark.parametrize(
        'chemical_symbols, atomic_numbers, formulas',
        [
            (
                ['H', 'H', 'O'],
                [1, 1, 8],
                ['H2O', 'H2O', 'H2O', 'H2O', 'A2B'],
            ),
            (
                ['O', 'O', 'O', 'O', 'La', 'Cu', 'Cu'],
                [8, 8, 8, 8, 57, 29, 29],
                ['LaCu2O4', 'Cu2LaO4', 'LaCu2O4', 'Cu2LaO4', 'A4B2C'],
            ),
            (
                ['O', 'La', 'As', 'Fe', 'C'],
                [8, 57, 33, 26, 6],
                ['CAsFeLaO', 'AsCFeLaO', 'LaFeCAsO', 'CAsFeLaO', 'ABCDE'],
            ),
        ],
    )
    def test_chemical_formula(self, chemical_symbols, atomic_numbers, formulas):
        """
        Test the `ChemicalFormula` normalization if a sibling `AtomicCell` is created, and thus the `Formula` class can be used.
        """
        atomic_cell = get_template_atomic_cell(
            chemical_symbols=chemical_symbols, atomic_numbers=atomic_numbers
        )
        chemical_formula = ChemicalFormula()
        model_system = ModelSystem(chemical_formula=chemical_formula)
        model_system.atomic_cell.append(atomic_cell)
        chemical_formula.normalize(None, logger)
        for index, name in enumerate(
            ['descriptive', 'reduced', 'iupac', 'hill', 'anonymous']
        ):
            assert getattr(chemical_formula, name) == formulas[index]

    @pytest.mark.parametrize(
        'positions, pbc, system_type, dimensionality',
        [
            (
                [[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]],
                None,
                'molecule / cluster',
                0,
            ),
            (
                [[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]],
                [False, False, False],
                'molecule / cluster',
                0,
            ),
            (
                [[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]],
                [True, False, False],
                '1D',
                1,
            ),
            (
                [[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]],
                [True, True, False],
                '2D',
                2,
            ),
            (
                [[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]],
                [True, True, True],
                'bulk',
                3,
            ),
        ],
    )
    def test_system_type_and_dimensionality(
        self, positions, pbc, system_type, dimensionality
    ):
        """
        Test the `ModelSystem` normalization of `type` and `dimensionality` from `AtomicCell`.
        """
        atomic_cell = get_template_atomic_cell(
            positions=positions, periodic_boundary_conditions=pbc
        )
        ase_atoms = atomic_cell.to_ase_atoms(logger)
        model_system = ModelSystem()
        model_system.atomic_cell.append(atomic_cell)
        (
            resolved_system_type,
            resolved_dimensionality,
        ) = model_system.resolve_system_type_and_dimensionality(ase_atoms, logger)
        assert resolved_system_type == system_type
        assert resolved_dimensionality == dimensionality

    def test_symmetry(self):
        """
        Test the `Symmetry` normalization from a sibling `AtomicCell` section.
        """
        atomic_cell = get_template_atomic_cell(
            periodic_boundary_conditions=[True, True, True]
        )
        assert (
            np.isclose(
                atomic_cell.lattice_vectors.to('angstrom').magnitude,
                np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            )
        ).all()
        symmetry = Symmetry()
        primitive, conventional = symmetry.resolve_bulk_symmetry(atomic_cell, logger)
        assert symmetry.bravais_lattice == 'hR'
        assert symmetry.hall_symbol == '-R 3 2"'
        assert symmetry.point_group_symbol == '-3m'
        assert symmetry.space_group_number == 166
        assert symmetry.space_group_symbol == 'R-3m'
        assert primitive.type == 'primitive'
        assert primitive.periodic_boundary_conditions == [False, False, False]
        assert (
            np.isclose(
                primitive.lattice_vectors.to('angstrom').magnitude,
                np.array(
                    [
                        [7.07106781e-01, 4.08248290e-01, 5.77350269e-01],
                        [-7.07106781e-01, 4.08248290e-01, 5.77350269e-01],
                        [1.08392265e-17, -8.16496581e-01, 5.77350269e-01],
                    ]
                ),
            )
        ).all()
        assert conventional.type == 'conventional'
        assert (
            np.isclose(
                conventional.lattice_vectors.to('angstrom').magnitude,
                np.array(
                    [
                        [1.41421356, 0.0, 0.0],
                        [-0.70710678, 1.22474487, 0.0],
                        [0.0, 0.0, 1.73205081],
                    ]
                ),
            )
        ).all()

    def test_no_representative(self):
        """
        Test the normalization of a `ModelSystem` is not run if it is not representative.
        """
        model_system = ModelSystem(is_representative=False)
        model_system.normalize(None, logger)
        assert model_system.type is None
        assert model_system.dimensionality is None

    def test_empty_atomic_cell(self):
        """
        Test the normalization of a `ModelSystem` is not run if it has no `AtomicCell` child section.
        """
        model_system = ModelSystem(is_representative=True)
        model_system.normalize(None, logger)
        assert model_system.type is None
        assert model_system.dimensionality is None

    def test_normalize(self):
        """
        Test the full normalization of a representative `ModelSystem`.
        """
        atomic_cell = get_template_atomic_cell(
            periodic_boundary_conditions=[True, True, True]
        )
        model_system = ModelSystem(is_representative=True)
        model_system.atomic_cell.append(atomic_cell)
        model_system.normalize(None, logger)
        # Basic quantities assertions
        assert model_system.type == 'bulk'
        assert model_system.dimensionality == 3
        # AtomicCell
        assert len(model_system.atomic_cell) == 3
        assert model_system.atomic_cell[0].type == 'original'
        assert model_system.atomic_cell[1].type == 'primitive'
        assert model_system.atomic_cell[2].type == 'conventional'
        # Symmetry
        assert len(model_system.symmetry) == 1
        assert model_system.symmetry[0].bravais_lattice == 'hR'
        assert model_system.symmetry[0].atomic_cell_ref == model_system.atomic_cell[2]
        # ChemicalFormula
        assert model_system.chemical_formula.descriptive == 'H2O'
        # ElementalComposition
        assert len(model_system.elemental_composition) == 2
        assert model_system.elemental_composition[0].element == 'H'
        assert np.isclose(model_system.elemental_composition[0].atomic_fraction, 2 / 3)
        assert model_system.elemental_composition[1].element == 'O'
        assert np.isclose(model_system.elemental_composition[1].atomic_fraction, 1 / 3)
