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

from nomad.units import ureg

from . import logger

from nomad_simulations.model_system import (
    GeometricSpace,
    Cell,
    AtomicCell,
    Symmetry,
    ChemicalFormula,
    ModelSystem,
)
from nomad_simulations.atoms_state import AtomsState


class TestAtomicCell:
    @pytest.mark.parametrize(
        'atoms, atomic_numbers, formula, lattice_vectors, positions, periodic_boundary_conditions',
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
        atoms,
        atomic_numbers,
        formula,
        lattice_vectors,
        positions,
        periodic_boundary_conditions,
    ):
        # Define the atomic cell
        atomic_cell = AtomicCell()
        if lattice_vectors:
            atomic_cell.lattice_vectors = lattice_vectors * ureg('angstrom')
        if positions:
            atomic_cell.positions = positions * ureg('angstrom')
        if periodic_boundary_conditions:
            atomic_cell.periodic_boundary_conditions = periodic_boundary_conditions

        # Add the elements information
        for index, atom in enumerate(atoms):
            atom_state = AtomsState()
            setattr(atom_state, 'chemical_symbol', atom)
            atomic_number = atom_state.resolve_atomic_number(logger)
            assert atomic_number == atomic_numbers[index]
            atom_state.atomic_number = atomic_number
            atomic_cell.atoms_state.append(atom_state)

        # Test `to_ase_atoms` function
        ase_atoms = atomic_cell.to_ase_atoms(logger)
        if not atoms or not positions or len(atoms) != len(positions):
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
