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

from . import logger

from nomad.units import ureg

from nomad_simulations import Simulation
from nomad_simulations.model_system import ModelSystem, AtomicCell
from nomad_simulations.atoms_state import AtomsState
from nomad_simulations.outputs import Outputs
from nomad_simulations.properties import (
    SpectralProfile,
    ElectronicDensityOfStates,
    XASSpectra,
)
from nomad_simulations.variables import Temperature, Energy2 as Energy


class TestSpectralProfile:
    """
    Test the `SpectralProfile` class defined in `properties/spectral_profile.py`.
    """

    def test_is_valid_spectral_profile(self):
        """
        Test the `is_valid_spectral_profile` method.
        """
        spectral_profile = SpectralProfile(
            variables=[Energy(grid_points=[-1, 0, 1] * ureg.joule)]
        )
        spectral_profile.value = [1.5, 0, 0.8]
        assert spectral_profile.is_valid_spectral_profile()
        spectral_profile.value = [2, 0, -4]
        assert not spectral_profile.is_valid_spectral_profile()


class TestElectronicDensityOfStates:
    """
    Test the `ElectronicDensityOfStates` class defined in `properties/spectral_profile.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `ElectronicDensityOfStates` class.
        """
        electronic_dos = ElectronicDensityOfStates()
        assert (
            electronic_dos.iri
            == 'http://fairmat-nfdi.eu/taxonomy/ElectronicDensityOfStates'
        )
        assert electronic_dos.name == 'ElectronicDensityOfStates'
        assert electronic_dos.rank == []

    def test_get_energy_points(self):
        """
        Test the `_get_energy_points` method.
        """
        electronic_dos = ElectronicDensityOfStates()
        electronic_dos.variables = [
            Temperature(grid_points=list(range(-3, 4)) * ureg.kelvin)
        ]
        assert electronic_dos._get_energy_points(logger) is None
        electronic_dos.variables.append(
            Energy(grid_points=list(range(-3, 4)) * ureg.joule)
        )
        energies = electronic_dos._get_energy_points(logger)
        assert (energies.magnitude == np.array([-3, -2, -1, 0, 1, 2, 3])).all()

    def test_resolve_energies_origin(self):
        """
        Test the `resolve_energies_origin` method.
        """
        # ! add test when `ElectronicEigenvalues` is implemented
        pass

    def test_resolve_normalization_factor(self, simulation_electronic_dos: Simulation):
        """
        Test the `resolve_normalization_factor` method.
        """
        simulation = Simulation()
        outputs = Outputs()
        # We only used the `simulation_electronic_dos` fixture to get the `ElectronicDensityOfStates` to test missing refs
        electronic_dos = simulation_electronic_dos.outputs[0].electronic_dos[0]
        electronic_dos.energies_origin = 0.5 * ureg.joule
        outputs.electronic_dos.append(electronic_dos)
        simulation.outputs.append(outputs)

        # No `model_system_ref`
        assert electronic_dos.resolve_normalization_factor(logger) is None

        # No `model_system_ref.cell`
        model_system = ModelSystem()
        simulation.model_system.append(model_system)
        outputs.model_system_ref = simulation.model_system[0]
        assert electronic_dos.resolve_normalization_factor(logger) is None

        # No `model_system_ref.cell.atoms_state`
        atomic_cell = AtomicCell(
            type='original', positions=[[0, 0, 0], [0.5, 0.5, 0.5]] * ureg.meter
        )
        model_system.cell.append(atomic_cell)
        assert electronic_dos.resolve_normalization_factor(logger) is None

        # Adding the required `model_system_ref` sections and quantities
        atoms_state = [
            AtomsState(chemical_symbol='Ga'),
            AtomsState(chemical_symbol='As'),
        ]
        for atom in atoms_state:
            atom.resolve_chemical_symbol_and_number(logger)
        atomic_cell.atoms_state = atoms_state
        # Non spin-polarized
        normalization_factor = electronic_dos.resolve_normalization_factor(logger)
        assert np.isclose(normalization_factor, 0.015625)
        # Spin-polarized
        electronic_dos.spin_channel = 0
        normalization_factor_spin_polarized = (
            electronic_dos.resolve_normalization_factor(logger)
        )
        assert np.isclose(
            normalization_factor_spin_polarized, 0.5 * normalization_factor
        )

    def test_extract_band_gap(self):
        """
        Test the `extract_band_gap` method.
        """
        # ! add test when `ElectronicEigenvalues` is implemented
        pass

    def test_resolve_pdos_name(self, simulation_electronic_dos: Simulation):
        """
        Test the `resolve_pdos_name` method.
        """
        # Get projected DOSProfile from the simulation fixture
        projected_dos = (
            simulation_electronic_dos.outputs[0].electronic_dos[0].projected_dos
        )
        assert len(projected_dos) == 3
        pdos_names = ['orbital s Ga', 'orbital px As', 'orbital py As']
        for i, pdos in enumerate(projected_dos):
            name = pdos.resolve_pdos_name(logger)
            assert name == pdos_names[i]

    def test_extract_projected_dos(self, simulation_electronic_dos: Simulation):
        """
        Test the `extract_projected_dos` method.
        """
        # Get Outputs and ElectronicDensityOfStates from the simulation fixture
        outputs = simulation_electronic_dos.outputs[0]
        electronic_dos = outputs.electronic_dos[0]

        # Initial tests for the passed `projected_dos` (only orbital PDOS)
        assert len(electronic_dos.projected_dos) == 3  # only orbital projected DOS
        orbital_projected = electronic_dos.extract_projected_dos('orbital', logger)
        atom_projected = electronic_dos.extract_projected_dos('atom', logger)
        assert len(orbital_projected) == 3 and len(atom_projected) == 0
        orbital_projected_names = [orb_pdos.name for orb_pdos in orbital_projected]
        assert orbital_projected_names == [
            'orbital s Ga',
            'orbital px As',
            'orbital py As',
        ]
        assert (
            orbital_projected[0].entity_ref
            == outputs.model_system_ref.cell[0].atoms_state[0].orbitals_state[0]
        )  # orbital `s` in `Ga` atom
        assert (
            orbital_projected[1].entity_ref
            == outputs.model_system_ref.cell[0].atoms_state[1].orbitals_state[0]
        )  # orbital `px` in `As` atom
        assert (
            orbital_projected[2].entity_ref
            == outputs.model_system_ref.cell[0].atoms_state[1].orbitals_state[1]
        )  # orbital `py` in `As` atom

        orbital_projected = electronic_dos.extract_projected_dos('orbital', logger)
        atom_projected = electronic_dos.extract_projected_dos('atom', logger)
        assert len(orbital_projected) == 3 and len(atom_projected) == 0

    @pytest.mark.parametrize(
        'value, result',
        [
            (None, [1.5, 1.2, 0, 0, 0, 0.8, 1.3]),
            ([1.5, 1.2, 0, 0, 0, 0.8, 1.3], [1.5, 1.2, 0, 0, 0, 0.8, 1.3]),
            ([30.5, 1.2, 0, 0, 0, 0.8, 1.3], [30.5, 1.2, 0, 0, 0, 0.8, 1.3]),
        ],
    )
    def test_generate_from_pdos(
        self,
        simulation_electronic_dos: Simulation,
        value: Optional[List[float]],
        result: List[float],
    ):
        """
        Test the `generate_from_projected_dos` method.
        """
        # Get Outputs and ElectronicDensityOfStates from the simulation fixture
        outputs = simulation_electronic_dos.outputs[0]
        electronic_dos = outputs.electronic_dos[0]

        # Add `value`
        if value is not None:
            electronic_dos.value = value * ureg('1/joule')

        val = electronic_dos.generate_from_projected_dos(logger)
        assert (val.magnitude == result).all()

        # Testing orbital + atom projected DOS (3 orbitals + 2 atoms PDOS)
        assert len(electronic_dos.projected_dos) == 5
        orbital_projected = electronic_dos.extract_projected_dos('orbital', logger)
        atom_projected = electronic_dos.extract_projected_dos('atom', logger)
        assert len(orbital_projected) == 3 and len(atom_projected) == 2
        atom_projected_names = [atom_pdos.name for atom_pdos in atom_projected]
        assert atom_projected_names == ['atom Ga', 'atom As']
        assert (
            atom_projected[0].entity_ref
            == outputs.model_system_ref.cell[0].atoms_state[0]
        )  # `Ga` atom
        assert (
            atom_projected[1].entity_ref
            == outputs.model_system_ref.cell[0].atoms_state[1]
        )  # `As` atom

    def test_normalize(self):
        """
        Test the `normalize` method.
        """
        # ! add test when `ElectronicEigenvalues` is implemented
        pass


class TestXASSpectra:
    """
    Test the `XASSpectra` class defined in `properties/spectral_profile.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `XASSpectra` class.
        """
        xas_spectra = XASSpectra()
        assert xas_spectra.iri is None  # Add iri when available
        assert xas_spectra.name == 'XASSpectra'
        assert xas_spectra.rank == []

    @pytest.mark.parametrize(
        'xanes_energies, exafs_energies, xas_values',
        [
            (None, None, None),
            ([0, 1, 2], None, None),
            (None, [3, 4, 5], None),
            ([0, 1, 2], [3, 4, 5], [0.5, 0.1, 0.3, 0.2, 0.4, 0.6]),
        ],
    )
    def test_generate_from_contributions(
        self,
        xanes_energies: Optional[List[float]],
        exafs_energies: Optional[List[float]],
        xas_values: Optional[List[float]],
    ):
        """
        Test the `generate_from_contributions` method.
        """
        # ! extend this test
        xas_spectra = XASSpectra()
        if xanes_energies is not None:
            xanes_spectra = SpectralProfile()
            xanes_spectra.variables = [Energy(grid_points=xanes_energies * ureg.joule)]
            xanes_spectra.value = [0.5, 0.1, 0.3]
            xas_spectra.xanes_spectra = xanes_spectra
        if exafs_energies is not None:
            exafs_spectra = SpectralProfile()
            exafs_spectra.variables = [Energy(grid_points=exafs_energies * ureg.joule)]
            exafs_spectra.value = [0.2, 0.4, 0.6]
            xas_spectra.exafs_spectra = exafs_spectra
        xas_spectra.generate_from_contributions(logger)
        if xas_spectra.value is not None:
            assert (xas_spectra.value == xas_values).all()
        else:
            assert xas_spectra.value == xas_values
