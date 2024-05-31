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

from nomad.datamodel import EntryArchive

from . import logger
from .conftest import generate_atomic_cell

from nomad_simulations.model_system import (
    Symmetry,
    ChemicalFormula,
    ModelSystem,
    AtomicCell,
    AtomsState
)
from nomad_simulations.general import Simulation


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
        chemical_symbols: List[str],
        atomic_numbers: List[int],
        formula: str,
        lattice_vectors: List[List[float]],
        positions: List[List[float]],
        periodic_boundary_conditions: List[bool],
    ):
        """
        Test the creation of `ase.Atoms` from `AtomicCell`.
        """
        atomic_cell = generate_atomic_cell(
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
        chemical_symbols: List[str],
        atomic_numbers: List[int],
        lattice_vectors: List[List[float]],
        positions: List[List[float]],
        vectors_results: List[Optional[float]],
        angles_results: List[Optional[float]],
        volume: Optional[float],
    ):
        """
        Test the `GeometricSpace` quantities normalization from `AtomicCell`.
        """
        pbc = [False, False, False]
        atomic_cell = generate_atomic_cell(
            lattice_vectors,
            positions,
            pbc,
            chemical_symbols,
            atomic_numbers,
        )

        # Get `GeometricSpace` quantities via normalization of `AtomicCell`
        atomic_cell.normalize(EntryArchive(), logger)
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
        chemical_formula.normalize(EntryArchive(), logger)
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
    def test_chemical_formula(
        self,
        chemical_symbols: List[str],
        atomic_numbers: List[int],
        formulas: List[str],
    ):
        """
        Test the `ChemicalFormula` normalization if a sibling `AtomicCell` is created, and thus the `Formula` class can be used.
        """
        atomic_cell = generate_atomic_cell(
            chemical_symbols=chemical_symbols, atomic_numbers=atomic_numbers
        )
        chemical_formula = ChemicalFormula()
        model_system = ModelSystem(chemical_formula=chemical_formula)
        model_system.cell.append(atomic_cell)
        chemical_formula.normalize(EntryArchive(), logger)
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
        self,
        positions: List[List[float]],
        pbc: Optional[List[bool]],
        system_type: str,
        dimensionality: int,
    ):
        """
        Test the `ModelSystem` normalization of `type` and `dimensionality` from `AtomicCell`.
        """
        atomic_cell = generate_atomic_cell(
            positions=positions, periodic_boundary_conditions=pbc
        )
        ase_atoms = atomic_cell.to_ase_atoms(logger)
        model_system = ModelSystem()
        model_system.cell.append(atomic_cell)
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
        atomic_cell = generate_atomic_cell(
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
        model_system.normalize(EntryArchive(), logger)
        assert model_system.type is None
        assert model_system.dimensionality is None

    def test_empty_atomic_cell(self):
        """
        Test the normalization of a `ModelSystem` is not run if it has no `AtomicCell` child section.
        """
        model_system = ModelSystem(is_representative=True)
        model_system.normalize(EntryArchive(), logger)
        assert model_system.type is None
        assert model_system.dimensionality is None

    def test_normalize(self):
        """
        Test the full normalization of a representative `ModelSystem`.
        """
        atomic_cell = generate_atomic_cell(
            periodic_boundary_conditions=[True, True, True]
        )
        model_system = ModelSystem(is_representative=True)
        model_system.cell.append(atomic_cell)
        model_system.normalize(EntryArchive(), logger)
        # Basic quantities assertions
        assert model_system.type == 'bulk'
        assert model_system.dimensionality == 3
        # AtomicCell
        assert len(model_system.cell) == 3
        assert model_system.cell[0].type == 'original'
        assert model_system.cell[1].type == 'primitive'
        assert model_system.cell[2].type == 'conventional'
        # Symmetry
        assert len(model_system.symmetry) == 1
        assert model_system.symmetry[0].bravais_lattice == 'hR'
        assert model_system.symmetry[0].atomic_cell_ref == model_system.cell[2]
        # ChemicalFormula
        assert model_system.chemical_formula.descriptive == 'H2O'
        # ElementalComposition
        assert len(model_system.elemental_composition) == 2
        assert model_system.elemental_composition[0].element == 'H'
        assert np.isclose(model_system.elemental_composition[0].atomic_fraction, 2 / 3)
        assert model_system.elemental_composition[1].element == 'O'
        assert np.isclose(model_system.elemental_composition[1].atomic_fraction, 1 / 3)


    # @pytest.mark.parametrize(
    #     'mol_label_list, n_mol_list, atom_labels_list, composition_formula_list',
    #     [
    #         (
    #             ['H20'],
    #             [3],
    #             [['H', 'O', 'O']],
    #             ['group_H20(1)', 'H20(3)', 'H(1)O(2)', 'H(1)O(2)', 'H(1)O(2)']
    #         ), # pure system
    #         (
    #             ['H20', 'Methane'],
    #             [5, 2],
    #             [['H', 'O', 'O'], ['C', 'H', 'H', 'H', 'H']],
    #             ['group_H20(1)group_Methane(1)', 'H20(5)', 'H(1)O(2)', 'H(1)O(2)', 'H(1)O(2)', 'H(1)O(2)', 'H(1)O(2)', 'Methane(2)', 'C(1)H(4)', 'C(1)H(4)']
    #         ), # binary mixture
    #     ],
    # )
    # def test_system_hierarchy_for_molecules(
    #     self,
    #     mol_label_list: List[str],
    #     n_mol_list: List[int],
    #     atom_labels_list: List[str],
    #     composition_formula_list: List[str]
    # ):
    #     """
    #     Test the `ModelSystem` normalization of 'composition_formula' for atoms and molecules.
    #     """
    #     #? Does it make sense to test the setting of branch_label or branch_depth?
    #     model_system = ModelSystem(is_representative=True)
    #     model_system.branch_label = 'Total System'
    #     model_system.branch_depth = 0
    #     atomic_cell = AtomicCell()
    #     model_system.cell.append(atomic_cell)
    #     model_system.atom_indices = []
    #     for (mol_label, n_mol, atom_labels) in zip(mol_label_list, n_mol_list, atom_labels_list):
    #         # Create a branch in the hierarchy for this molecule type
    #         model_system_mol_group = ModelSystem(branch_label='group' + mol_label)
    #         model_system_mol_group.atom_indices = []
    #         model_system_mol_group.branch_label = f"group_{mol_label}"
    #         model_system_mol_group.branch_depth = 1
    #         model_system.model_system.append(model_system_mol_group)
    #         for _ in range(n_mol):
    #             # Create a branch in the hierarchy for this molecule
    #             model_system_mol = ModelSystem(branch_label=mol_label)
    #             model_system_mol.branch_label = mol_label
    #             model_system_mol.branch_depth = 2
    #             model_system_mol_group.model_system.append(model_system_mol)
    #             # add the corresponding atoms to the global atom list
    #             for atom_label in atom_labels:
    #                 atomic_cell.atoms_state.append(AtomsState(chemical_symbol = atom_label))
    #             n_atoms = len(atomic_cell.atoms_state)
    #             atom_indices = np.arange(n_atoms - len(atom_labels), n_atoms)
    #             model_system_mol.atom_indices = atom_indices
    #             model_system_mol_group.atom_indices = np.append(model_system_mol_group.atom_indices, atom_indices)
    #             model_system.atom_indices = np.append(model_system.atom_indices, atom_indices)

    #     model_system.normalize(EntryArchive(), logger)

    #     assert model_system.composition_formula == composition_formula_list[0]
    #     ctr_comp = 1
    #     def get_system_recurs(sec_system, ctr_comp):
    #         for sys in sec_system:
    #             assert sys.composition_formula == composition_formula_list[ctr_comp]
    #             ctr_comp += 1
    #             sec_subsystem = sys.model_system
    #             if sec_subsystem:
    #                 ctr_comp = get_system_recurs(sec_subsystem, ctr_comp)
    #         return ctr_comp

    #     get_system_recurs(model_system.model_system, ctr_comp)

    @pytest.mark.parametrize(
        'mol_label_list, n_mol_list, atom_labels_list, composition_formula_list',
        [
            (
                ['H20'],
                [3],
                [['H', 'O', 'O']],
                ['group_H20(1)', 'H20(3)', 'H(1)O(2)', 'H(1)O(2)', 'H(1)O(2)']
            ), # pure system
            (
                ['H20', 'Methane'],
                [5, 2],
                [['H', 'O', 'O'], ['C', 'H', 'H', 'H', 'H']],
                ['group_H20(1)group_Methane(1)', 'H20(5)', 'H(1)O(2)', 'H(1)O(2)', 'H(1)O(2)', 'H(1)O(2)', 'H(1)O(2)', 'Methane(2)', 'C(1)H(4)', 'C(1)H(4)']
            ), # binary mixture
        ],
    )
    def test_system_hierarchy_for_molecules(
        self,
        mol_label_list: List[str],
        n_mol_list: List[int],
        atom_labels_list: List[str],
        composition_formula_list: List[str]
    ):
        """
        Test the `ModelSystem` normalization of 'composition_formula' for atoms and molecules.
        """
        simulation = Simulation()
        #? Does it make sense to test the setting of branch_label or branch_depth?
        model_system = ModelSystem(is_representative=True)
        simulation.model_system.append(model_system)
        model_system.branch_label = 'Total System'
        model_system.branch_depth = 0
        atomic_cell = AtomicCell()
        model_system.cell.append(atomic_cell)
        model_system.atom_indices = []
        for (mol_label, n_mol, atom_labels) in zip(mol_label_list, n_mol_list, atom_labels_list):
            # Create a branch in the hierarchy for this molecule type
            model_system_mol_group = ModelSystem(branch_label='group' + mol_label)
            model_system_mol_group.atom_indices = []
            model_system_mol_group.branch_label = f"group_{mol_label}"
            model_system_mol_group.branch_depth = 1
            model_system.model_system.append(model_system_mol_group)
            for _ in range(n_mol):
                # Create a branch in the hierarchy for this molecule
                model_system_mol = ModelSystem(branch_label=mol_label)
                model_system_mol.branch_label = mol_label
                model_system_mol.branch_depth = 2
                model_system_mol_group.model_system.append(model_system_mol)
                # add the corresponding atoms to the global atom list
                for atom_label in atom_labels:
                    atomic_cell.atoms_state.append(AtomsState(chemical_symbol = atom_label))
                n_atoms = len(atomic_cell.atoms_state)
                atom_indices = np.arange(n_atoms - len(atom_labels), n_atoms)
                model_system_mol.atom_indices = atom_indices
                model_system_mol_group.atom_indices = np.append(model_system_mol_group.atom_indices, atom_indices)
                model_system.atom_indices = np.append(model_system.atom_indices, atom_indices)

        # model_system.normalize(EntryArchive(), logger)
        simulation.normalize(EntryArchive(), logger)

        assert model_system.composition_formula == composition_formula_list[0]
        ctr_comp = 1
        def get_system_recurs(sec_system, ctr_comp):
            for sys in sec_system:
                assert sys.composition_formula == composition_formula_list[ctr_comp]
                ctr_comp += 1
                sec_subsystem = sys.model_system
                if sec_subsystem:
                    ctr_comp = get_system_recurs(sec_subsystem, ctr_comp)
            return ctr_comp

        get_system_recurs(model_system.model_system, ctr_comp)