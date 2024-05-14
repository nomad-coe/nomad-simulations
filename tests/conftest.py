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

import os
import pytest
from typing import List, Optional

from nomad.units import ureg
from nomad.datamodel import EntryArchive

from . import logger

from nomad_simulations import Simulation
from nomad_simulations.model_system import ModelSystem, AtomicCell
from nomad_simulations.atoms_state import AtomsState, OrbitalsState
from nomad_simulations.model_method import ModelMethod
from nomad_simulations.numerical_settings import SelfConsistency
from nomad_simulations.outputs import Outputs, SCFOutputs
from nomad_simulations.variables import Energy2 as Energy
from nomad_simulations.properties import (
    ElectronicBandGap,
    DOSProfile,
    ElectronicDensityOfStates,
)

if os.getenv('_PYTEST_RAISE', '0') != '0':

    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call):
        raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo):
        raise excinfo.value


def generate_simulation(
    model_system: Optional[ModelSystem] = None,
    model_method: Optional[ModelMethod] = None,
    outputs: Optional[Outputs] = None,
) -> Simulation:
    """
    Generate a `Simulation` section with the main sub-sections, `ModelSystem`, `ModelMethod`, and `Outputs`. If `ModelSystem`
    and `Outputs` are set, then it adds `ModelSystem` as a reference in `Outputs`.
    """
    simulation = Simulation()
    if model_method is not None:
        simulation.model_method.append(model_method)
    if model_system is not None:
        simulation.model_system.append(model_system)
    if outputs is not None:
        simulation.outputs.append(outputs)
        outputs.model_system_ref = model_system
    return simulation


def generate_model_system(
    type: str = 'original',
    positions: List[List[float]] = [[0, 0, 0], [0.5, 0.5, 0.5]],
    chemical_symbols: List[str] = ['Ga', 'As'],
    orbitals_symbols: List[List[str]] = [['s'], ['px', 'py']],
) -> Optional[ModelSystem]:
    """
    Generate a `ModelSystem` section with the given parameters.
    """
    if len(chemical_symbols) != len(orbitals_symbols):
        return None

    model_system = ModelSystem()
    atomic_cell = AtomicCell(type=type, positions=positions * ureg.meter)
    model_system.cell.append(atomic_cell)

    # Add atoms_state to the model_system
    atoms_state = []
    for element, orbitals in zip(chemical_symbols, orbitals_symbols):
        orbitals_state = []
        for orbital in orbitals:
            orbitals_state.append(
                OrbitalsState(
                    l_quantum_symbol=orbital[0], ml_quantum_symbol=orbital[1:]
                )
            )  # TODO add this split setter as part of the `OrbitalsState` methods
        atom_state = AtomsState(chemical_symbol=element, orbitals_state=orbitals_state)
        # and obtain the atomic number for each AtomsState
        atom_state.normalize(EntryArchive(), logger)
        atoms_state.append(atom_state)
    atomic_cell.atoms_state = atoms_state
    return model_system


def generate_atomic_cell(
    lattice_vectors: List = [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    positions=None,
    periodic_boundary_conditions=None,
    chemical_symbols: List = ['H', 'H', 'O'],
    atomic_numbers: List = [1, 1, 8],
) -> AtomicCell:
    """
    Generate an `AtomicCell` section with the given parameters.
    """
    # Define positions if not provided
    if positions is None and chemical_symbols is not None:
        n_atoms = len(chemical_symbols)
        positions = [[i / n_atoms, i / n_atoms, i / n_atoms] for i in range(n_atoms)]
    # Define periodic boundary conditions if not provided
    if periodic_boundary_conditions is None:
        periodic_boundary_conditions = [False, False, False]

    # Define the atomic cell
    atomic_cell = AtomicCell()
    if lattice_vectors:
        atomic_cell.lattice_vectors = lattice_vectors * ureg('angstrom')
    if positions:
        atomic_cell.positions = positions * ureg('angstrom')
    if periodic_boundary_conditions:
        atomic_cell.periodic_boundary_conditions = periodic_boundary_conditions

    # Add the elements information
    for index, atom in enumerate(chemical_symbols):
        atom_state = AtomsState()
        setattr(atom_state, 'chemical_symbol', atom)
        atomic_number = atom_state.resolve_atomic_number(logger)
        assert atomic_number == atomic_numbers[index]
        atom_state.atomic_number = atomic_number
        atomic_cell.atoms_state.append(atom_state)

    return atomic_cell


def generate_scf_electronic_band_gap_template(
    threshold_change: float = 1e-3,
) -> SCFOutputs:
    """
    Generate a `SCFOutputs` section with a template for the electronic_band_gap property.
    """
    scf_outputs = SCFOutputs()
    # Define a list of scf_steps with values of the total energy like [1, 1.1, 1.11, 1.111, etc],
    # such that the difference between one step and the next one decreases a factor of 10.
    n_scf_steps = 5
    for i in range(1, n_scf_steps):
        value = 1 + sum([1 / (10**j) for j in range(1, i + 1)])
        scf_step = Outputs(
            electronic_band_gap=[ElectronicBandGap(value=value * ureg.joule)]
        )
        scf_outputs.scf_steps.append(scf_step)
    # Add a SCF calculated PhysicalProperty
    scf_outputs.electronic_band_gap.append(ElectronicBandGap(value=value * ureg.joule))
    # and a `SelfConsistency` ref section
    scf_params = SelfConsistency(
        threshold_change=threshold_change, threshold_change_unit='joule'
    )
    scf_outputs.electronic_band_gap[0].self_consistency_ref = scf_params
    return scf_outputs


def generate_simulation_electronic_dos(
    energy_points: List[int] = [-3, -2, -1, 0, 1, 2, 3],
) -> Simulation:
    """
    Generate a `Simulation` section with an `ElectronicDensityOfStates` section under `Outputs`. It uses
    the template of the model_system created with the `generate_model_system` function.
    """
    # Create the `Simulation` section to make refs work
    model_system = generate_model_system()
    outputs = Outputs()
    simulation = generate_simulation(model_system=model_system, outputs=outputs)

    # Populating the `ElectronicDensityOfStates` section
    variables_energy = [Energy(points=energy_points * ureg.joule)]
    electronic_dos = ElectronicDensityOfStates(variables=variables_energy)
    outputs.electronic_dos.append(electronic_dos)
    # electronic_dos.value = total_dos * ureg('1/joule')
    orbital_s_Ga_pdos = DOSProfile(
        variables=variables_energy,
        entity_ref=model_system.cell[0].atoms_state[0].orbitals_state[0],
    )
    orbital_px_As_pdos = DOSProfile(
        variables=variables_energy,
        entity_ref=model_system.cell[0].atoms_state[1].orbitals_state[0],
    )
    orbital_py_As_pdos = DOSProfile(
        variables=variables_energy,
        entity_ref=model_system.cell[0].atoms_state[1].orbitals_state[1],
    )
    orbital_s_Ga_pdos.value = [0.2, 0.5, 0, 0, 0, 0.0, 0.0] * ureg('1/joule')
    orbital_px_As_pdos.value = [1.0, 0.2, 0, 0, 0, 0.3, 0.0] * ureg('1/joule')
    orbital_py_As_pdos.value = [0.3, 0.5, 0, 0, 0, 0.5, 1.3] * ureg('1/joule')
    electronic_dos.projected_dos = [
        orbital_s_Ga_pdos,
        orbital_px_As_pdos,
        orbital_py_As_pdos,
    ]
    return simulation


@pytest.fixture(scope='session')
def model_system() -> ModelSystem:
    return generate_model_system()


@pytest.fixture(scope='session')
def atomic_cell() -> AtomicCell:
    return generate_atomic_cell()


@pytest.fixture(scope='session')
def scf_electronic_band_gap() -> SCFOutputs:
    return generate_scf_electronic_band_gap_template()


@pytest.fixture(scope='session')
def simulation_electronic_dos() -> Simulation:
    return generate_simulation_electronic_dos()
