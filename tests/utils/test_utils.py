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
