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
from nomad.datamodel.datamodel import EntryArchive

from nomad_simulations.schema_packages.general import Simulation
from nomad_simulations.schema_packages.model_method import ModelMethod
from nomad_simulations.schema_packages.model_system import (
    AtomicCell,
    ModelSystem,
    Symmetry,
)
from nomad_simulations.schema_packages.outputs import Outputs
from nomad_simulations.schema_packages.utils import (
    extract_all_simulation_subsections,
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
    assert get_sibling_section(sibling_section, 'cell', logger).type == section.type
    assert get_sibling_section(section, 'symmetry', logger, index_sibling=2) is None
    section2 = AtomicCell(type='primitive')
    parent_section.cell.append(section2)
    assert (
        get_sibling_section(sibling_section, 'cell', logger, index_sibling=0).type
        == 'original'
    )
    assert (
        get_sibling_section(sibling_section, 'cell', logger, index_sibling=0).type
        == section.type
    )
    assert (
        get_sibling_section(sibling_section, 'cell', logger, index_sibling=1).type
        == section2.type
    )
    assert (
        get_sibling_section(sibling_section, 'cell', logger, index_sibling=1).type
        == 'primitive'
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
    'archive, subsection_indices, result',
    [
        # no data section
        (
            EntryArchive(),
            [0, -1, -1],
            [None, None, None],
        ),
        # no subsections
        (
            EntryArchive(data=Simulation()),
            [0, -1, -1],
            [None, None, None],
        ),
        # no model_method and outputs
        (
            EntryArchive(data=Simulation(model_system=[ModelSystem()])),
            [0, -1, -1],
            [None, None, None],
        ),
        # no outputs
        (
            EntryArchive(
                data=Simulation(
                    model_system=[ModelSystem()], model_method=[ModelMethod()]
                )
            ),
            [0, -1, -1],
            [None, None, None],
        ),
        # all subsections
        (
            EntryArchive(
                data=Simulation(
                    model_system=[ModelSystem()],
                    model_method=[ModelMethod()],
                    outputs=[Outputs()],
                )
            ),
            [0, -1, -1],
            [ModelSystem(), ModelMethod(), Outputs()],
        ),
        # wrong index for model_system
        (
            EntryArchive(
                data=Simulation(
                    model_system=[ModelSystem()],
                    model_method=[ModelMethod()],
                    outputs=[Outputs()],
                )
            ),
            [2, -1, -1],
            [None, None, None],
        ),
    ],
)
def test_extract_all_simulation_subsections(
    archive: EntryArchive, subsection_indices: list, result: list
):
    """
    Test the `extract_all_simulation_subsections` utility function.
    """
    system, method, output = extract_all_simulation_subsections(
        archive=archive,
        i_system=subsection_indices[0],
        i_method=subsection_indices[1],
        i_output=subsection_indices[2],
    )
    if result[0] is not None:
        assert (
            isinstance(system, ModelSystem)
            and isinstance(method, ModelMethod)
            and isinstance(output, Outputs)
        )
    else:
        assert system == result[0] and method == result[1] and output == result[2]
