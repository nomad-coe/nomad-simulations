from typing import Optional

import ase
import numpy as np
import pytest
from nomad.datamodel import EntryArchive

from nomad_simulations.schema_packages.atoms_state import AtomsState
from nomad_simulations.schema_packages.model_system import (
    AtomicCell,
    Cell,
    ChemicalFormula,
    ModelSystem,
    Symmetry,
)

from . import logger
from .conftest import generate_atomic_cell


class TestAtomicCell:
    """
    Test the `AtomicCell`, `Cell` and `GeometricSpace` classes defined in model_system.py
    """

    @pytest.mark.parametrize(
        'cell_1, cell_2, result',
        [
            (Cell(), None, {'lt': False, 'gt': False, 'eq': False}),  # one cell is None
            # (Cell(), Cell(), False),  # both cells are empty
            # (
            #    Cell(positions=[[1, 0, 0]]),
            #    Cell(),
            #    False,
            # ),  # one cell has positions, the other is empty
            (
                Cell(positions=[[1, 0, 0]]),
                Cell(positions=[[2, 0, 0]]),
                {'lt': False, 'gt': False, 'eq': False},
            ),  # position vectors are treated as the fundamental set elements
            (
                Cell(positions=[[1, 0, 0], [0, 1, 0]]),
                Cell(positions=[[1, 0, 0]]),
                {'lt': False, 'gt': True, 'eq': False},
            ),  # one is a subset of the other
            (
                Cell(positions=[[1, 0, 0]]),
                Cell(positions=[[1, 0, 0], [0, 1, 0]]),
                {'lt': True, 'gt': False, 'eq': False},
            ),  # one is a subset of the other
            (
                Cell(positions=[[1, 0, 0], [0, 1, 0]]),
                Cell(positions=[[1, 0, 0], [0, -1, 0]]),
                {'lt': False, 'gt': False, 'eq': False},
            ),  # different positions
            (
                Cell(positions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                Cell(positions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                {'lt': False, 'gt': False, 'eq': True},
            ),  # same ordered positions
            (
                Cell(positions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                Cell(positions=[[1, 0, 0], [0, 0, 1], [0, 1, 0]]),
                {'lt': False, 'gt': False, 'eq': True},
            ),  # different ordered positions but same cell
            # (
            #    AtomicCell(positions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            #    Cell(positions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            #    False,
            # ),  # one atomic cell and another cell (missing chemical symbols)
            # (
            #    AtomicCell(positions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            #    AtomicCell(positions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            #    False,
            # ),  # missing chemical symbols
            # ND: the comparison will now return an error here
            #     handling a case that should be resolved by the normalizer falls outside its scope
            (
                AtomicCell(
                    positions=[[1, 0, 0]],
                    atoms_state=[
                        AtomsState(chemical_symbol='O'),
                    ],
                ),
                AtomicCell(
                    positions=[[1, 0, 0]],
                    atoms_state=[
                        AtomsState(chemical_symbol='H'),
                    ],
                ),
                {'lt': False, 'gt': False, 'eq': False},
            ),  # chemical symbols are treated as the fundamental set elements
            (
                AtomicCell(
                    positions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    atoms_state=[
                        AtomsState(chemical_symbol='H'),
                        AtomsState(chemical_symbol='H'),
                        AtomsState(chemical_symbol='O'),
                    ],
                ),
                AtomicCell(
                    positions=[[1, 0, 0], [0, 1, 0]],
                    atoms_state=[
                        AtomsState(chemical_symbol='H'),
                        AtomsState(chemical_symbol='H'),
                    ],
                ),
                {'lt': False, 'gt': True, 'eq': False},
            ),  # one is a subset of the other
            (
                AtomicCell(
                    positions=[[1, 0, 0], [0, 1, 0]],
                    atoms_state=[
                        AtomsState(chemical_symbol='H'),
                        AtomsState(chemical_symbol='H'),
                    ],
                ),
                AtomicCell(
                    positions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    atoms_state=[
                        AtomsState(chemical_symbol='H'),
                        AtomsState(chemical_symbol='H'),
                        AtomsState(chemical_symbol='O'),
                    ],
                ),
                {'lt': True, 'gt': False, 'eq': False},
            ),  # one is a subset of the other
            (
                AtomicCell(
                    positions=[[1, 0, 0], [0, 1, 0]],
                    atoms_state=[
                        AtomsState(chemical_symbol='H'),
                        AtomsState(chemical_symbol='O'),
                    ],
                ),
                AtomicCell(
                    positions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    atoms_state=[
                        AtomsState(chemical_symbol='H'),
                        AtomsState(chemical_symbol='H'),
                        AtomsState(chemical_symbol='O'),
                    ],
                ),
                {'lt': False, 'gt': False, 'eq': False},
            ),
            (
                AtomicCell(
                    positions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    atoms_state=[
                        AtomsState(chemical_symbol='H'),
                        AtomsState(chemical_symbol='H'),
                        AtomsState(chemical_symbol='O'),
                    ],
                ),
                AtomicCell(
                    positions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    atoms_state=[
                        AtomsState(chemical_symbol='H'),
                        AtomsState(chemical_symbol='H'),
                        AtomsState(chemical_symbol='O'),
                    ],
                ),
                {'lt': False, 'gt': False, 'eq': True},
            ),  # same ordered positions and chemical symbols
            (
                AtomicCell(
                    positions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    atoms_state=[
                        AtomsState(chemical_symbol='H'),
                        AtomsState(chemical_symbol='H'),
                        AtomsState(chemical_symbol='O'),
                    ],
                ),
                AtomicCell(
                    positions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    atoms_state=[
                        AtomsState(chemical_symbol='H'),
                        AtomsState(chemical_symbol='Cu'),
                        AtomsState(chemical_symbol='O'),
                    ],
                ),
                {'lt': False, 'gt': False, 'eq': False},
            ),  # same ordered positions but different chemical symbols
            (
                AtomicCell(
                    positions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    atoms_state=[
                        AtomsState(chemical_symbol='H'),
                        AtomsState(chemical_symbol='H'),
                        AtomsState(chemical_symbol='O'),
                    ],
                ),
                AtomicCell(
                    positions=[[1, 0, 0], [0, 0, 1], [0, 1, 0]],
                    atoms_state=[
                        AtomsState(chemical_symbol='H'),
                        AtomsState(chemical_symbol='O'),
                        AtomsState(chemical_symbol='H'),
                    ],
                ),
                {'lt': False, 'gt': False, 'eq': True},
            ),  # same position-symbol map, different overall order
            (
                AtomicCell(
                    positions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    atoms_state=[
                        AtomsState(chemical_symbol='H'),
                        AtomsState(chemical_symbol='H'),
                        AtomsState(chemical_symbol='O'),
                    ],
                ),
                AtomicCell(
                    positions=[[1, 0, 0], [0, 0, 1], [0, 1, 0]],
                    atoms_state=[
                        AtomsState(chemical_symbol='H'),
                        AtomsState(chemical_symbol='H'),
                        AtomsState(chemical_symbol='O'),
                    ],
                ),
                {'lt': False, 'gt': False, 'eq': False},
            ),  # different position-symbol map
        ],
    )
    def test_partial_order(self, cell_1: Cell, cell_2: Cell, result: dict[str, bool]):
        """
        Test the comparison operators of `Cell` and `AtomicCell`.
        """
        assert cell_1.is_lt_cell(cell_2) == result['lt']
        assert cell_1.is_gt_cell(cell_2) == result['gt']
        assert cell_1.is_le_cell(cell_2) == (result['lt'] or result['eq'])
        assert cell_1.is_ge_cell(cell_2) == (result['gt'] or result['eq'])
        assert cell_1.is_equal_cell(cell_2) == result['eq']
        assert cell_1.is_ne_cell(cell_2) == (not result['eq'])

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
    def test_to_ase_atoms(
        self,
        chemical_symbols: list[str],
        atomic_numbers: list[int],
        formula: str,
        lattice_vectors: list[list[float]],
        positions: list[list[float]],
        periodic_boundary_conditions: list[bool],
    ):
        """
        Test the creation of `ase.Atoms` from `AtomicCell`.

        Args:
            chemical_symbols (list[str]): List of chemical symbols.
            atomic_numbers (list[int]): List of atomic numbers.
            formula (str): Chemical formula.
            lattice_vectors (list[list[float]]): Lattice vectors.
            positions (list[list[float]]): Atomic positions.
            periodic_boundary_conditions (list[bool]): Periodic boundary conditions.
        """
        atomic_cell = generate_atomic_cell(
            lattice_vectors=lattice_vectors,
            positions=positions,
            periodic_boundary_conditions=periodic_boundary_conditions,
            chemical_symbols=chemical_symbols,
            atomic_numbers=atomic_numbers,
        )

        # Test `to_ase_atoms` function
        ase_atoms = atomic_cell.to_ase_atoms(logger=logger)
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
        'ase_atoms, chemical_symbols, pbc, lattice_vectors, positions',
        [
            (
                ase.Atoms(),
                [],
                [False, False, False],
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                None,
            ),
            (
                ase.Atoms(symbols='CO'),
                ['C', 'O'],
                [False, False, False],
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [[0, 0, 0], [0, 0, 0]],
            ),
            (
                ase.Atoms(symbols='CO', pbc=True),
                ['C', 'O'],
                [True, True, True],
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [[0, 0, 0], [0, 0, 0]],
            ),
            (
                ase.Atoms(symbols='CO', positions=[[0, 0, 0], [0, 0, 1.1]]),
                ['C', 'O'],
                [False, False, False],
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [[0, 0, 0], [0, 0, 1.1]],
            ),
            (
                ase.Atoms(
                    symbols='Au', positions=[[0, 5, 5]], cell=[2.9, 5, 5], pbc=[1, 0, 0]
                ),
                ['Au'],
                [True, False, False],
                [[2.9, 0, 0], [0, 5, 0], [0, 0, 5]],
                [[0, 5, 5]],
            ),
        ],
    )
    def test_from_ase_atoms(
        self,
        ase_atoms: ase.Atoms,
        chemical_symbols: list[str],
        pbc: list[bool],
        lattice_vectors: list,
        positions: list,
    ):
        atomic_cell = AtomicCell()
        atomic_cell.from_ase_atoms(ase_atoms=ase_atoms, logger=logger)
        assert atomic_cell.get_chemical_symbols(logger=logger) == chemical_symbols
        assert atomic_cell.periodic_boundary_conditions == pbc
        assert (
            atomic_cell.lattice_vectors.to('angstrom').magnitude
            == np.array(lattice_vectors)
        ).all()
        if positions is None:
            assert atomic_cell.positions is None
        else:
            assert (
                atomic_cell.positions.to('angstrom').magnitude == np.array(positions)
            ).all()

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
        chemical_symbols: list[str],
        atomic_numbers: list[int],
        lattice_vectors: list[list[float]],
        positions: list[list[float]],
        vectors_results: list[Optional[float]],
        angles_results: list[Optional[float]],
        volume: Optional[float],
    ):
        """
        Test the `GeometricSpace` quantities normalization from `AtomicCell`.

        Args:
            chemical_symbols (list[str]): List of chemical symbols.
            atomic_numbers (list[int]): List of atomic numbers.
            lattice_vectors (list[list[float]]): Lattice vectors.
            positions (list[list[float]]): Atomic positions.
            vectors_results (list[Optional[float]]): Expected lengths of cell vectors.
            angles_results (list[Optional[float]]): Expected angles between cell vectors.
            volume (Optional[float]): Expected volume of the cell.
        """
        atomic_cell = generate_atomic_cell(
            lattice_vectors=lattice_vectors,
            positions=positions,
            chemical_symbols=chemical_symbols,
            atomic_numbers=atomic_numbers,
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
        chemical_symbols: list[str],
        atomic_numbers: list[int],
        formulas: list[str],
    ):
        """
        Test the `ChemicalFormula` normalization if a sibling `AtomicCell` is created, and thus the `Formula` class can be used.

        Args:
            chemical_symbols (list[str]): List of chemical symbols.
            atomic_numbers (list[int]): List of atomic numbers.
            formulas (list[str]): List of expected formulas.
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
        positions: list[list[float]],
        pbc: Optional[list[bool]],
        system_type: str,
        dimensionality: int,
    ):
        """
        Test the `ModelSystem` normalization of `type` and `dimensionality` from `AtomicCell`.

        Args:
            positions (list[list[float]]): Atomic positions.
            pbc (Optional[list[bool]]): Periodic boundary conditions.
            system_type (str): Expected system type.
            dimensionality (int): Expected dimensionality.
        """
        atomic_cell = generate_atomic_cell(
            positions=positions, periodic_boundary_conditions=pbc
        )
        ase_atoms = atomic_cell.to_ase_atoms(logger=logger)
        model_system = ModelSystem()
        model_system.cell.append(atomic_cell)
        (
            resolved_system_type,
            resolved_dimensionality,
        ) = model_system.resolve_system_type_and_dimensionality(
            ase_atoms=ase_atoms, logger=logger
        )
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
        primitive, conventional = symmetry.resolve_bulk_symmetry(
            original_atomic_cell=atomic_cell, logger=logger
        )
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
