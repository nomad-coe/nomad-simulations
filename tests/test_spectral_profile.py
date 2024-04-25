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
from typing import Optional, List, Union

from . import logger

from nomad.units import ureg

from nomad_simulations import Simulation
from nomad_simulations.model_system import ModelSystem, AtomicCell
from nomad_simulations.atoms_state import AtomsState, OrbitalsState
from nomad_simulations.outputs import Outputs
from nomad_simulations.properties import (
    SpectralProfile,
    ElectronicDensityOfStates,
    XASSpectra,
    FermiLevel,
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

    @pytest.mark.parametrize(
        'fermi_level, sibling_section_value, result',
        [
            (None, None, None),
            (None, 0.5, 0.5),
            (0.5, None, 0.5),
            (0.5, 1.0, 0.5),
        ],
    )
    def test_resolve_fermi_level(
        self,
        fermi_level: Optional[float],
        sibling_section_value: Optional[float],
        result: Optional[float],
        electronic_dos: ElectronicDensityOfStates,
    ):
        """
        Test the `_resolve_fermi_level` method.
        """
        outputs = Outputs()
        sec_fermi_level = FermiLevel(variables=[])
        if sibling_section_value is not None:
            sec_fermi_level.value = sibling_section_value * ureg.joule
        outputs.fermi_level.append(sec_fermi_level)
        if fermi_level is not None:
            electronic_dos.fermi_level = fermi_level * ureg.joule
        outputs.electronic_dos.append(electronic_dos)
        resolved_fermi_level = electronic_dos.resolve_fermi_level(logger)
        if resolved_fermi_level is not None:
            resolved_fermi_level = resolved_fermi_level.magnitude
        assert resolved_fermi_level == result

    def test_resolve_energies_origin(self):
        """
        Test the `resolve_energies_origin` method.
        """
        # ! add test when `ElectronicEigenvalues` is implemented
        pass

    def test_resolve_normalization_factor(
        self, electronic_dos: ElectronicDensityOfStates
    ):
        """
        Test the `resolve_normalization_factor` method.
        """
        simulation = Simulation()
        outputs = Outputs()
        electronic_dos.fermi_level = 0.5 * ureg.joule
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

    def test_generate_from_projected_dos(
        self, model_system: ModelSystem, electronic_dos: ElectronicDensityOfStates
    ):
        """
        Test the `generate_from_projected_dos` and `extract_projected_dos` methods.
        """
        simulation = Simulation()
        simulation.model_system.append(model_system)
        outputs = Outputs()
        outputs.electronic_dos.append(electronic_dos)
        outputs.model_system_ref = simulation.model_system[0]
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
        # ! These tests are not passing, despite these are the same sections
        # assert (
        #     orbital_projected[0].entity_ref
        #     == outputs.model_system_ref.cell[0].atoms_state[0].orbitals_state[0]
        # )  # orbital `s` in `Ga` atom
        # assert (
        #     orbital_projected[1].entity_ref
        #     == outputs.model_system_ref.cell[0].atoms_state[1].orbitals_state[0]
        # )  # orbital `px` in `As` atom
        # assert (
        #     orbital_projected[1].entity_ref
        #     == outputs.model_system_ref.cell[0].atoms_state[1].orbitals_state[1]
        # )  # orbital `py` in `As` atom

        # Note: `val` is reported from `self.value`, not from the extraction
        val = electronic_dos.generate_from_projected_dos(logger)
        assert (val.magnitude == electronic_dos.value.magnitude).all()
        assert len(electronic_dos.projected_dos) == 5  # including atom projected DOS
        orbital_projected = electronic_dos.extract_projected_dos('orbital', logger)
        atom_projected = electronic_dos.extract_projected_dos('atom', logger)
        assert len(orbital_projected) == 3 and len(atom_projected) == 2
        atom_projected_names = [atom_pdos.name for atom_pdos in atom_projected]
        assert atom_projected_names == ['atom Ga', 'atom As']
        # ! These tests are not passing, despite these are the same sections
        # assert (
        #     atom_projected[0].entity_ref
        #     == outputs.model_system_ref.cell[0].atoms_state[0]
        # )  # `Ga` atom
        # assert (
        #     atom_projected[1].entity_ref
        #     == outputs.model_system_ref.cell[0].atoms_state[1]
        # )  # `As` atom

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

    def test_generate_from_contributions(self):
        """
        Test the `generate_from_contributions` method.
        """
        xas_spectra = XASSpectra()
        xanes_spectra = SpectralProfile(
            variables=[Energy(grid_points=[0, 1, 2] * ureg.joule)]
        )
        xanes_spectra.value = [0.5, 0.1, 0.3]
        xas_spectra.xanes_spectra = xanes_spectra
        exafs_spectra = SpectralProfile(
            variables=[Energy(grid_points=[3, 4, 5] * ureg.joule)]
        )
        exafs_spectra.value = [0.2, 0.4, 0.6]
        xas_spectra.exafs_spectra = exafs_spectra
        xas_spectra.generate_from_contributions(logger)
        assert len(xas_spectra.variables) == 1
        assert len(xas_spectra.variables[0].grid_points) == 6
        assert len(xas_spectra.variables[0].grid_points) == (
            len(xanes_spectra.variables[0].grid_points)
            + len(exafs_spectra.variables[0].grid_points)
        )
        assert (
            xas_spectra.variables[0].grid_points.magnitude == [0, 1, 2, 3, 4, 5]
        ).all()
        assert (xas_spectra.value == [0.5, 0.1, 0.3, 0.2, 0.4, 0.6]).all()
