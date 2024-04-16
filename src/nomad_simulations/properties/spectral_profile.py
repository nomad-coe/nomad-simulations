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

import numpy as np
from structlog.stdlib import BoundLogger
from typing import Optional
import pint

from nomad import config
from nomad.units import ureg
from nomad.metainfo import (
    Quantity,
    SubSection,
    MEnum,
    Section,
    Context,
    JSON,
)

from ..utils import get_sibling_section
from ..physical_property import PhysicalProperty
from ..variables import Energy2 as Energy
from .band_gap import ElectronicBandGap


class SpectralProfile(PhysicalProperty):
    """
    A base section used to define the spectral profile.
    """

    value = Quantity(
        type=np.float64,
        description="""
        The value of the intensities of a spectral profile in arbitrary units.
        """,
    )

    # TODO implement this in PhysicalProperty (similar to `value.shape`)
    # value_units = Quantity(
    #     type=str,
    #     description="""
    #     Unit using the pint UnitRegistry() notation for the `value`. Example, if the spectra `is_derived`
    #     from the imaginary part of the dielectric function, `value` are `'F/m'`.
    #     """,
    # )

    def __init__(
        self, m_def: Section = None, m_context: Context = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.rank = []  # ? Is this here or in the attrs instantiation better?

    def is_valid_spectral_profile(self) -> bool:
        """
        Check if the spectral profile is valid, i.e., if all `value` are defined positive.

        Returns:
            (bool): True if the spectral profile is valid, False otherwise.
        """
        if (self.value < 0.0).any():
            return False
        return True

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        if self.is_valid_spectral_profile() is False:
            logger.error(
                'Invalid negative intensities found: could not validate spectral profile.'
            )
            return


class ElectronicDensityOfStates(SpectralProfile):
    """
    Number of electronic states accessible for the charges per energy and per volume.
    """

    iri = 'http://fairmat-nfdi.eu/taxonomy/ElectronicDensityOfStates'

    spin_channel = Quantity(
        type=np.int32,
        description="""
        Spin channel of the corresponding electronic DOS. It can take values of 0 or 1.
        """,
    )

    fermi_level = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        The Fermi level is the highest occupied energy level at zero temperature. For insulators and semiconductors,
        some codes have some difficulty in determining the Fermi level, and it is often set to the middle of the band gap,
        at the top of the valence band, or even in the bottom of the conduction band. This quantity is extracted
        from `FermiLevel.value` property.
        """,
    )

    energies_origin = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        Energy level denoting the origin along the energy axis, used for comparison and visualization. It is
        defined as the `ElectronicEigenvalues.highest_occupied` and does not necessarily coincide with `fermi_level`.
        """,
    )

    normalization_factor = Quantity(
        type=np.float64,
        description="""
        Normalization factor for electronic DOS to get a cell-independent intensive DOS. The cell-independent
        intensive DOS is as the integral from the lowest (most negative) energy to the `fermi_level` for a neutrally
        charged system (i.e., the sum of `AtomsState.charge` is zero).
        """,
    )

    value = Quantity(
        type=np.float64,
        unit='1/joule',
        description="""
        The value of the electronic DOS.
        """,
    )

    value_integrated = Quantity(
        type=np.float64,
        description="""
        The cumulative intensities integrated from from the lowest (most negative) energy to the `fermi_level`.
        """,
    )

    projected_dos = SubSection(
        sub_section=SpectralProfile.m_def,
        repeats=True,
        description="""
        Projected DOS. It can be species- (same elements in the unit cell), atom- (different elements in the unit cell),
        or orbital-projected. These can be calculated in a cascade as:
            - If the total DOS is not present, we can sum all species-projected DOS to obtain it.
            - If the species-projected DOS is not present, we can sum all atom-projected DOS to obtain it.
            - If the atom-projected DOS is not present, we can sum all orbital-projected DOS to obtain it.
        The `name` of the projected DOS is set to the species, atom, or orbital name from the corresponding `atoms_state_ref`
        or `orbitals_state_ref`.
        """,
    )

    def __init__(
        self, m_def: Section = None, m_context: Context = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.name = self.m_def.name

    def _check_energy_variables(self, logger: BoundLogger) -> Optional[pint.Quantity]:
        """
        Check if the required `Energy` variable is present in the `variables`. If so, it returns
        the grid points of the `Energy` variable.

        Args:
            logger (BoundLogger): The logger to log messages.

        Returns:
            (Optional[pint.Quantity]): The grid points of the `Energy` variable.
        """
        for var in self.variables:
            if isinstance(var, Energy):
                return var.grid_points
        logger.error(
            'The required `Energy` variable is not present in the `variables`.'
        )
        return None

    def _check_spin_polarized(self, logger: BoundLogger) -> bool:
        """
        Check if the simulation `is_spin_polarized`.

        Args:
            logger (BoundLogger): The logger to log messages.

        Returns:
            (bool): True if the simulation is spin-polarized, False otherwise.
        """
        # TODO use this in `ElectronicBandGap` too
        model_method = get_sibling_section(
            section=self, sibling_section_name='model_method', logger=logger
        )
        return (
            True
            if model_method is not None and model_method.is_spin_polarized
            else False
        )

    def resolve_fermi_level(self, logger: BoundLogger) -> Optional[pint.Quantity]:
        """
        Resolve the Fermi level from a sibling `FermiLevel` section, if present.

        Args:
            logger (BoundLogger): The logger to log messages.

        Returns:
            (Optional[pint.Quantity]): The resolved Fermi level.
        """
        fermi_level_value = None
        if self.fermi_level is None:
            fermi_level = get_sibling_section(
                section=self, sibling_section_name='fermi_level', logger=logger
            )  # we consider `index_sibling` to be 0
            if fermi_level is not None:
                fermi_level_value = fermi_level.value
        return fermi_level_value

    def resolve_energies_origin(
        self,
        energies: pint.Quantity,
        fermi_level: Optional[pint.Quantity],
        logger: BoundLogger,
    ) -> Optional[pint.Quantity]:
        """
        Resolve the origin of reference for the energies from the sibling `ElectronicEigenvalues` section and its
        `highest_occupied` level, or if this does not exist, from the `fermi_level` value as extracted from `resolve_fermi_level()`.

        Args:
            fermi_level (Optional[pint.Quantity]): The resolved Fermi level.
            energies (pint.Quantity): The grid points of the `Energy` variable.
            logger (BoundLogger): The logger to log messages.

        Returns:
            (Optional[pint.Quantity]): The resolved origin of reference for the energies.
        """
        # ! We need schema for `ElectronicEigenvalues` to store `highest_occupied` and `lowest_occupied`

        # Check if the variables contain more than one variable (different than Energy)
        # ? Is this correct or should be use the index of energies to extract the proper shape element in `self.value` being used for `dos_values`?
        if len(self.variables) > 1:
            logger.warning(
                'The ElectronicDensityOfStates section contains more than one variable. We cannot extract the energy reference.'
            )
            return None

        # Extract the `ElectronicEigenvalues` section to get the `highest_occupied` and `lowest_occupied` energies
        # TODO implement once `ElectronicEigenvalues` is in the schema
        eigenvalues = get_sibling_section(
            section=self, sibling_section_name='electronic_eigenvalues', logger=logger
        )  # we consider `index_sibling` to be 0
        highest_occupied_energy = (
            eigenvalues.highest_occupied if eigenvalues is not None else None
        )
        lowest_occupied_energy = (
            eigenvalues.lowest_occupied if eigenvalues is not None else None
        )
        # and set defaults for `highest_occupied_energy` and `lowest_occupied_energy` in `m_cache`
        if highest_occupied_energy is not None:
            self.m_cache['highest_occupied_energy'] = highest_occupied_energy
        if lowest_occupied_energy is not None:
            self.m_cache['lowest_occupied_energy'] = lowest_occupied_energy

        # Set thresholds for the energies and values
        energy_threshold = config.normalize.band_structure_energy_tolerance
        value_threshold = 1e-8  # The DOS value that is considered to be zero

        # Check that the closest `energies` to the energy reference is not too far away.
        # If it is very far away, normalization may be very inaccurate and we do not report it.
        dos_values = self.value.magnitude
        eref = highest_occupied_energy if fermi_level is None else fermi_level
        fermi_idx = (np.abs(energies - eref)).argmin()
        fermi_energy_closest = energies[fermi_idx]
        distance = np.abs(fermi_energy_closest - eref)
        single_peak_fermi = False
        if distance.magnitude <= energy_threshold:
            # See if there are zero values close below the energy reference.
            idx = fermi_idx
            idx_descend = fermi_idx
            while True:
                try:
                    value = dos_values[idx]
                    energy_distance = np.abs(eref - energies[idx])
                except IndexError:
                    break
                if energy_distance.magnitude > energy_threshold:
                    break
                if value <= value_threshold:
                    idx_descend = idx
                    break
                idx -= 1

            # See if there are zero values close above the fermi energy.
            idx = fermi_idx
            idx_ascend = fermi_idx
            while True:
                try:
                    value = dos_values[idx]
                    energy_distance = np.abs(eref - energies[idx])
                except IndexError:
                    break
                if energy_distance.magnitude > energy_threshold:
                    break
                if value <= value_threshold:
                    idx_ascend = idx
                    break
                idx += 1

            # If there is a single peak at fermi energy, no
            # search needs to be performed.
            if idx_ascend != fermi_idx and idx_descend != fermi_idx:
                self.m_cache['highest_occupied_energy'] = fermi_energy_closest
                self.m_cache['lowest_occupied_energy'] = fermi_energy_closest
                single_peak_fermi = True

            if not single_peak_fermi:
                # Look for highest occupied energy below the descend index
                idx = idx_descend
                while True:
                    try:
                        value = dos_values[idx]
                    except IndexError:
                        break
                    if value > value_threshold:
                        idx = idx if idx == idx_descend else idx + 1
                        self.m_cache['highest_occupied_energy'] = energies[idx]
                        break
                    idx -= 1
                # Look for lowest unoccupied energy above idx_ascend
                idx = idx_ascend
                while True:
                    try:
                        value = dos_values[idx]
                    except IndexError:
                        break
                    if value > value_threshold:
                        idx = idx if idx == idx_ascend else idx - 1
                        self.m_cache['highest_occupied_energy'] = energies[idx]
                        break
                    idx += 1

        # Return the `highest_occupied_energy` as the `energies_origin`, or the `fermi_level` if it is not None
        energies_origin = self.m_cache.get('highest_occupied_energy')
        if energies_origin is None:
            energies_origin = fermi_level
        return energies_origin

    def extract_band_gap(self) -> Optional[ElectronicBandGap]:
        """
        Extract the electronic band gap from the `highest_occupied_energy` and `lowest_occupied_energy` stored
        in `m_cache` from `resolve_energies_origin()`. If the difference of `highest_occupied_energy` and
        `lowest_occupied_energy` is negative, the band gap `value` is set to 0.0.

        Returns:
            (Optional[ElectronicBandGap]): The extracted electronic band gap section to be stored in `Outputs`.
        """
        band_gap = None
        homo = self.m_cache.get('highest_occupied_energy')
        lumo = self.m_cache.get('lowest_occupied_energy')
        if homo and lumo:
            band_gap = ElectronicBandGap()
            band_gap.is_derived = True
            band_gap.physical_property_ref = self

            if (homo - lumo).magnitude < 0:
                band_gap.value = 0.0
            else:
                band_gap.value = homo - lumo
        return band_gap

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        # Initial check to see if `variables` contains the required `Energy` variable
        energies = self._check_energy_variables(logger)
        if energies is None:
            return

        # Check if `is_spin_polarized` but `spin_channel` is not set
        if self._check_spin_polarized(logger) and self.spin_channel is None:
            logger.warning(
                'Spin-polarized calculation detected but the `spin_channel` is not set.'
            )

        # Resolve `fermi_level` from a sibling section with respect to `ElectronicDensityOfStates`
        self.fermi_level = self.resolve_fermi_level(logger)
        # and the `energies_origin` from the sibling `ElectronicEigenvalues` section
        self.energies_origin = self.resolve_energies_origin(
            energies, self.fermi_level, logger
        )
        if self.energies_origin is None:
            logger.info('Could not resolve the `energies_origin` for the DOS')

        # `ElectronicBandGap` extraction
        band_gap = self.extract_band_gap()
        if band_gap is not None:
            self.m_parent.electronic_band_gap.append(band_gap)


class XASSpectra(SpectralProfile):
    """
    X-ray Absorption Spectra (XAS).
    """

    xanes_spectra = SubSection(
        sub_section=SpectralProfile.m_def,
        description="""
        X-ray Absorption Near Edge Structure (XANES) spectra.
        """,
        repeats=False,
    )

    exafs_spectra = SubSection(
        sub_section=SpectralProfile.m_def,
        description="""
        Extended X-ray Absorption Fine Structure (EXAFS) spectra.
        """,
        repeats=False,
    )

    def __init__(
        self, m_def: Section = None, m_context: Context = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        # Set the name of the section
        self.name = self.m_def.name

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)
