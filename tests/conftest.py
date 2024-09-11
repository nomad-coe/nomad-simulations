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
from typing import Optional

import numpy as np
import pytest
from nomad.datamodel import EntryArchive
from nomad.units import ureg

from nomad_simulations.schema_packages.atoms_state import AtomsState, OrbitalsState
from nomad_simulations.schema_packages.general import Simulation
from nomad_simulations.schema_packages.model_method import ModelMethod
from nomad_simulations.schema_packages.model_system import AtomicCell, ModelSystem
from nomad_simulations.schema_packages.numerical_settings import (
    KLinePath as KLinePathSettings,
)
from nomad_simulations.schema_packages.numerical_settings import (
    KMesh as KMeshSettings,
)
from nomad_simulations.schema_packages.numerical_settings import (
    KSpace,
    SelfConsistency,
)
from nomad_simulations.schema_packages.outputs import Outputs, SCFOutputs
from nomad_simulations.schema_packages.properties import (
    DOSProfile,
    ElectronicBandGap,
    ElectronicDensityOfStates,
    ElectronicEigenvalues,
)
from nomad_simulations.schema_packages.variables import Energy2 as Energy
from nomad_simulations.schema_packages.variables import KLinePath

from . import logger

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
    system_type: str = 'bulk',
    positions: list[list[float]] = [[0, 0, 0], [0.5, 0.5, 0.5]],
    lattice_vectors: list[list[float]] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    chemical_symbols: list[str] = ['Ga', 'As'],
    orbitals_symbols: list[list[str]] = [['s'], ['px', 'py']],
    is_representative: bool = True,
    pbc: list[bool] = [False, False, False],
) -> Optional[ModelSystem]:
    """
    Generate a `ModelSystem` section with the given parameters.
    """
    if len(chemical_symbols) != len(orbitals_symbols):
        return None

    model_system = ModelSystem(type=system_type, is_representative=is_representative)
    atomic_cell = AtomicCell(
        type=type,
        positions=positions * ureg.angstrom,
        lattice_vectors=lattice_vectors * ureg.angstrom,
        periodic_boundary_conditions=pbc,
    )
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
    lattice_vectors: list[list[float]] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    positions: Optional[list] = None,
    periodic_boundary_conditions: list[bool] = [False, False, False],
    chemical_symbols: list[str] = ['H', 'H', 'O'],
    atomic_numbers: list[int] = [1, 1, 8],
) -> AtomicCell:
    """
    Generate an `AtomicCell` section with the given parameters.
    """
    # Define positions if not provided
    if positions is None and chemical_symbols is not None:
        n_atoms = len(chemical_symbols)
        positions = [[i / n_atoms, i / n_atoms, i / n_atoms] for i in range(n_atoms)]

    # Define the atomic cell
    atomic_cell = AtomicCell(periodic_boundary_conditions=periodic_boundary_conditions)
    if lattice_vectors:
        atomic_cell.lattice_vectors = lattice_vectors * ureg('angstrom')
    if positions:
        atomic_cell.positions = positions * ureg('angstrom')

    # Add the elements information
    for index, atom in enumerate(chemical_symbols):
        atom_state = AtomsState()
        setattr(atom_state, 'chemical_symbol', atom)
        atomic_number = atom_state.resolve_atomic_number(logger=logger)
        assert atomic_number == atomic_numbers[index]
        atom_state.atomic_number = atomic_number
        atomic_cell.atoms_state.append(atom_state)

    return atomic_cell


def generate_scf_electronic_band_gap_template(
    n_scf_steps: int = 5,
    threshold_change: Optional[float] = 1e-3,
    threshold_change_unit: Optional[str] = 'joule',
) -> SCFOutputs:
    """
    Generate a `SCFOutputs` section with a template for the electronic_band_gap property.
    """
    scf_outputs = SCFOutputs()
    # Define a list of scf_steps with values of the total energy like [1, 1.1, 1.11, 1.111, etc],
    # such that the difference between one step and the next one decreases a factor of 10.
    value = None
    for i in range(n_scf_steps):
        value = 1 + sum([1 / (10**j) for j in range(1, i + 2)])
        scf_step = Outputs(electronic_band_gaps=[ElectronicBandGap(value=value)])
        scf_outputs.scf_steps.append(scf_step)
    # Add a SCF calculated PhysicalProperty
    if value is not None:
        scf_outputs.electronic_band_gaps.append(ElectronicBandGap(value=value))
    else:
        scf_outputs.electronic_band_gaps.append(ElectronicBandGap())
    # and a `SelfConsistency` ref section
    if threshold_change is not None:
        model_method = ModelMethod(
            numerical_settings=[
                SelfConsistency(
                    threshold_change=threshold_change,
                    threshold_change_unit=threshold_change_unit,
                )
            ]
        )
        simulation = generate_simulation(model_method=model_method, outputs=scf_outputs)
        scf_outputs.electronic_band_gaps[
            0
        ].self_consistency_ref = simulation.model_method[0].numerical_settings[0]
    return scf_outputs


def generate_simulation_electronic_dos(
    energy_points: list[int] = [-3, -2, -1, 0, 1, 2, 3],
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


def generate_k_line_path(
    high_symmetry_path_names: list[str] = ['Gamma', 'X', 'Y', 'Gamma'],
    high_symmetry_path_values: list[list[float]] = [
        [0, 0, 0],
        [0.5, 0, 0],
        [0, 0.5, 0],
        [0, 0, 0],
    ],
) -> KLinePathSettings:
    return KLinePathSettings(
        high_symmetry_path_names=high_symmetry_path_names,
        high_symmetry_path_values=high_symmetry_path_values,
    )


def generate_k_space_simulation(
    system_type: str = 'bulk',
    is_representative: bool = True,
    positions: list[list[float]] = [[0, 0, 0], [0.5, 0.5, 0.5]],
    lattice_vectors: list[list[float]] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    chemical_symbols: list[str] = ['Ga', 'As'],
    orbitals_symbols: list[list[str]] = [['s'], ['px', 'py']],
    pbc: list[bool] = [False, False, False],
    reciprocal_lattice_vectors: Optional[list[list[float]]] = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ],
    high_symmetry_path_names: list[str] = ['Gamma', 'X', 'Y', 'Gamma'],
    high_symmetry_path_values: list[list[float]] = [
        [0, 0, 0],
        [0.5, 0, 0],
        [0, 0.5, 0],
        [0, 0, 0],
    ],
    klinepath_points: Optional[list[float]] = None,
    grid=[6, 6, 6],
) -> Simulation:
    model_system = generate_model_system(
        system_type=system_type,
        is_representative=is_representative,
        positions=positions,
        lattice_vectors=lattice_vectors,
        chemical_symbols=chemical_symbols,
        orbitals_symbols=orbitals_symbols,
        pbc=pbc,
    )
    k_space = KSpace()
    # adding `reciprocal_lattice_vectors`
    if reciprocal_lattice_vectors is not None:
        k_space.reciprocal_lattice_vectors = (
            2 * np.pi * np.array(reciprocal_lattice_vectors) / ureg.angstrom
        )
    # adding `KMeshSettings
    k_mesh = KMeshSettings(grid=grid)
    k_space.k_mesh.append(k_mesh)
    # adding `KLinePathSettings`
    k_line_path = KLinePathSettings(
        high_symmetry_path_names=high_symmetry_path_names,
        high_symmetry_path_values=high_symmetry_path_values,
    )
    if klinepath_points is not None:
        k_line_path.points = klinepath_points
    k_space.k_line_path = k_line_path
    # appending `KSpace` to `ModelMethod.numerical_settings`
    model_method = ModelMethod()
    model_method.numerical_settings.append(k_space)
    return generate_simulation(model_method=model_method, model_system=model_system)


def generate_electronic_eigenvalues(
    reciprocal_lattice_vectors: Optional[list[list[float]]] = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ],
    value: Optional[list] = [
        [3, -2],
        [3, 1],
        [4, -2],
        [5, -1],
        [4, 0],
        [2, 0],
        [2, 1],
        [4, -3],
    ],
    occupation: Optional[list] = [
        [0, 2],
        [0, 1],
        [0, 2],
        [0, 2],
        [0, 1.5],
        [0, 1.5],
        [0, 1],
        [0, 2],
    ],
    highest_occupied: Optional[float] = None,
    lowest_unoccupied: Optional[float] = None,
) -> ElectronicEigenvalues:
    """
    Generate an `ElectronicEigenvalues` section with the given parameters.
    """
    outputs = Outputs()
    k_space = KSpace(
        k_line_path=KLinePathSettings(
            points=[
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ]
        )
    )
    model_method = ModelMethod(numerical_settings=[k_space])
    if reciprocal_lattice_vectors is not None and len(reciprocal_lattice_vectors) > 0:
        k_space.reciprocal_lattice_vectors = reciprocal_lattice_vectors
    _ = generate_simulation(
        model_system=generate_model_system(),
        model_method=model_method,
        outputs=outputs,
    )
    electronic_eigenvalues = ElectronicEigenvalues(n_bands=2)
    outputs.electronic_eigenvalues.append(electronic_eigenvalues)
    electronic_eigenvalues.variables = [
        KLinePath(points=model_method.numerical_settings[0].k_line_path)
    ]
    if value is not None:
        electronic_eigenvalues.value = value
    if occupation is not None:
        electronic_eigenvalues.occupation = occupation
    electronic_eigenvalues.highest_occupied = highest_occupied
    electronic_eigenvalues.lowest_unoccupied = lowest_unoccupied
    return electronic_eigenvalues


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


@pytest.fixture(scope='session')
def k_line_path() -> KLinePathSettings:
    return generate_k_line_path()


@pytest.fixture(scope='session')
def k_space_simulation() -> Simulation:
    return generate_k_space_simulation()


refs_apw = [
    {
        'm_def': 'nomad_simulations.schema_packages.basis_set.BasisSetContainer',
    },
    {
        'm_def': 'nomad_simulations.schema_packages.basis_set.BasisSetContainer',
        'basis_set_components': [
            {
                'm_def': 'nomad_simulations.schema_packages.basis_set.APWPlaneWaveBasisSet',
                'cutoff_energy': 500.0,
            },
        ],
    },
    {
        'm_def': 'nomad_simulations.schema_packages.basis_set.BasisSetContainer',
        'basis_set_components': [
            {
                'm_def': 'nomad_simulations.schema_packages.basis_set.APWPlaneWaveBasisSet',
                'cutoff_energy': 500.0,
            },
            {
                'm_def': 'nomad_simulations.schema_packages.basis_set.MuffinTinRegion',
                'species_scope': ['/data/model_system/0/cell/0/atoms_state/0'],
                'radius': 1.0,
                'l_max': 2,
                'l_channels': [
                    {
                        'name': 0,
                        'orbitals': [
                            {
                                'm_def': 'nomad_simulations.schema_packages.basis_set.APWOrbital',
                                'energy_parameter': [0.0],
                                'differential_order': [0],
                            },
                        ],
                    },
                    {
                        'name': 1,
                        'orbitals': [
                            {
                                'm_def': 'nomad_simulations.schema_packages.basis_set.APWOrbital',
                                'energy_parameter': [0.0],
                                'differential_order': [0],
                            },
                        ],
                    },
                    {
                        'name': 2,
                        'orbitals': [
                            {
                                'm_def': 'nomad_simulations.schema_packages.basis_set.APWOrbital',
                                'energy_parameter': [0.0],
                                'differential_order': [0],
                            },
                        ],
                    },
                ],
            },
        ],
    },
]
