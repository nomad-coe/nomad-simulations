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

from nomad_simulations.schema_packages.atoms_state import AtomsState
from nomad_simulations.schema_packages.model_system import (
    AtomicCell,
    Cell,
    ModelSystem,
    Symmetry,
)
from nomad_simulations.schema_packages.utils import (
    check_simulation_cell,
    get_sibling_section,
    get_variables,
    is_not_representative,
)
from nomad_simulations.schema_packages.variables import Energy2 as Energy
from nomad_simulations.schema_packages.variables import Temperature

from . import logger


def test_get_sibling_section():
    """
    Test the `get_sibling_section` utility function.
    """
    parent_section = ModelSystem()
    section = AtomicCell(type='original')
    parent_section.cell.append(section)
    sibling_section = Symmetry()
    parent_section.symmetry.append(sibling_section)
    assert get_sibling_section(section, '', logger) is None
    assert get_sibling_section(section, 'symmetry', logger) == sibling_section
    assert get_sibling_section(sibling_section, 'cell', logger) == section
    assert get_sibling_section(section, 'symmetry', logger, index_sibling=2) is None
    section2 = AtomicCell(type='primitive')
    parent_section.cell.append(section2)
    assert (
        get_sibling_section(sibling_section, 'cell', logger, index_sibling=0) == section
    )
    assert (
        get_sibling_section(sibling_section, 'cell', logger, index_sibling=1)
        == section2
    )


def test_is_not_representative():
    """
    Test the `is_not_representative` utility function.
    """
    assert is_not_representative(None, logger) is None
    assert is_not_representative(ModelSystem(), logger)
    assert not is_not_representative(ModelSystem(is_representative=True), logger)


# ! Missing test for RusselSandersState (but this class will probably be deprecated)


@pytest.mark.parametrize(
    'variables, result, result_length',
    [
        (None, [], 0),
        ([], [], 0),
        ([Temperature()], [], 0),
        ([Temperature(), Energy(n_points=4)], [Energy(n_points=4)], 1),
        (
            [Temperature(), Energy(n_points=2), Energy(n_points=10)],
            [Energy(n_points=2), Energy(n_points=10)],
            2,
        ),
        # TODO add testing when we have variables which inherit from another variable
    ],
)
def test_get_variables(variables: list, result: list, result_length: int):
    """
    Test the `get_variables` utility function
    """
    energies = get_variables(variables, Energy)
    assert len(energies) == result_length
    for i, energy in enumerate(energies):  # asserting energies == result does not work
        assert energy.n_points == result[i].n_points


@pytest.mark.parametrize(
    'cell_1, cell_2, result',
    [
        (None, None, False),  # both are None
        (Cell(), None, False),  # one cell is None
        (Cell(), Cell(), False),  # both cells are empty
        (
            Cell(positions=[[1, 0, 0]]),
            Cell(),
            False,
        ),  # one cell has positions, the other is empty
        (
            Cell(positions=[[1, 0, 0], [0, 1, 0]]),
            Cell(positions=[[1, 0, 0]]),
            False,
        ),  # length mismatch
        (
            Cell(positions=[[1, 0, 0], [0, 1, 0]]),
            Cell(positions=[[1, 0, 0], [0, -1, 0]]),
            False,
        ),  # different positions
        (
            Cell(positions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            Cell(positions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            True,
        ),  # same ordered positions
        (
            Cell(positions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            Cell(positions=[[1, 0, 0], [0, 0, 1], [0, 1, 0]]),
            True,
        ),  # different ordered positions but same cell
        (
            AtomicCell(positions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            Cell(positions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            False,
        ),  # one atomic cell and another cell (missing chemical symbols)
        (
            AtomicCell(positions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            AtomicCell(positions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            False,
        ),  # missing chemical symbols
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
            True,
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
            False,
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
            True,
        ),  # different ordered positions but same chemical symbols
    ],
)
def test_check_simulation_cell(cell_1: Cell, cell_2: Cell, result: bool):
    """
    Test the `check_simulation_cell` utility function.
    """
    assert check_simulation_cell(cell_1=cell_1, cell_2=cell_2) == result
