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

from typing import TYPE_CHECKING, Optional

import numpy as np
import pint
from nomad.config import config
from nomad.metainfo import MEnum, Quantity, SubSection

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from nomad.metainfo import Context, Section
    from structlog.stdlib import BoundLogger

from nomad_simulations.schema_packages.atoms_state import AtomsState, OrbitalsState
from nomad_simulations.schema_packages.physical_property import PhysicalProperty
from nomad_simulations.schema_packages.properties.band_gap import ElectronicBandGap
from nomad_simulations.schema_packages.utils import get_sibling_section, get_variables
from nomad_simulations.schema_packages.variables import Energy2 as Energy

configuration = config.get_plugin_entry_point(
    'nomad_simulations.schema_packages:nomad_simulations_plugin'
)


class SpectralProfile(PhysicalProperty):
    """
    A base section used to define the spectral profile.
    """

    value = Quantity(
        type=np.float64,
        description="""
        The value of the intensities of a spectral profile in arbitrary units.
        """,
    )  # TODO check units and normalization_factor of DOS and Spectras and see whether they can be merged

    def __init__(
        self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.rank = []

    def is_valid_spectral_profile(self) -> bool:
        """
        Check if the spectral profile is valid, i.e., if all `value` are defined positive.

        Returns:
            (bool): True if the spectral profile is valid, False otherwise.
        """
        if (self.value < 0.0).any():
            return False
        return True

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

        if self.is_valid_spectral_profile() is False:
            logger.error(
                'Invalid negative intensities found: could not validate spectral profile.'
            )
            return


class DOSProfile(SpectralProfile):
    """
    A base section used to define the `value` of the `ElectronicDensityOfState` property. This is useful when containing
    contributions for `projected_dos` with the correct unit.
    """

    value = Quantity(
        type=np.float64,
        unit='1/joule',
        description="""
        The value of the electronic DOS.
        """,
    )

    def __init__(
        self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)

    def resolve_pdos_name(self, logger: 'BoundLogger') -> Optional[str]:
        """
        Resolve the `name` of the projected `DOSProfile` from the `entity_ref` section. This is resolved as:
            - `'atom X'` with 'X' being the chemical symbol for `AtomsState` references.
            -  `'orbital Y X'` with 'X' being the chemical symbol and 'Y' the orbital label for `OrbitalsState` references.

        Args:
            logger (BoundLogger): The logger to log messages.

        Returns:
            (Optional[str]): The resolved `name` of the projected DOS profile.
        """
        if self.entity_ref is None:
            logger.warning(
                'The `entity_ref` is not set for the DOS profile. Could not resolve the `name`.'
            )
            return None

        # Resolve the `name` from the `entity_ref`
        name = None
        if isinstance(self.entity_ref, AtomsState):
            name = f'atom {self.entity_ref.chemical_symbol}'
        elif isinstance(self.entity_ref, OrbitalsState):
            name = f'orbital {self.entity_ref.l_quantum_symbol}{self.entity_ref.ml_quantum_symbol} {self.entity_ref.m_parent.chemical_symbol}'
        return name

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

        # We resolve
        self.name = self.resolve_pdos_name(logger)


class ElectronicDensityOfStates(DOSProfile):
    """
    Number of electronic states accessible for the charges per energy and per volume.
    """

    # ! implement `iri` and `rank` as part of `m_def = Section()`
    iri = 'http://fairmat-nfdi.eu/taxonomy/ElectronicDensityOfStates'

    spin_channel = Quantity(
        type=np.int32,
        description="""
        Spin channel of the corresponding electronic DOS. It can take values of 0 or 1.
        """,
    )

    # TODO clarify the role of `energies_origin` once `ElectronicEigenvalues` is implemented
    energies_origin = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        Energy level denoting the origin along the energy axis, used for comparison and visualization. It is
        defined as the `ElectronicEigenvalues.highest_occupied_energy`.
        """,
    )

    normalization_factor = Quantity(
        type=np.float64,
        description="""
        Normalization factor for electronic DOS to get a cell-independent intensive DOS. The cell-independent
        intensive DOS is as the integral from the lowest (most negative) energy to the Fermi level for a neutrally
        charged system (i.e., the sum of `AtomsState.charge` is zero).
        """,
    )

    # ? Do we want to store the integrated value here os as part of an nomad-analysis tool? Check `dos_integrator.py` module in dos normalizer repository
    # value_integrated = Quantity(
    #     type=np.float64,
    #     description="""
    #     The cumulative intensities integrated from from the lowest (most negative) energy to the Fermi level.
    #     """,
    # )

    projected_dos = SubSection(
        sub_section=DOSProfile.m_def,
        repeats=True,
        description="""
        Projected DOS. It can be atom- (different elements in the unit cell) or orbital-projected. These can be calculated in a cascade as:
            - If the total DOS is not present, we sum all atom-projected DOS to obtain it.
            - If the atom-projected DOS is not present, we sum all orbital-projected DOS to obtain it.
        Note: the cover given by summing up contributions is not perfect, and will depend on the projection functions used.

        In `projected_dos`, `name` and `entity_ref` must be set in order for normalization to work:
            - The `entity_ref` is the `OrbitalsState` or `AtomsState` sections.
            - The `name` of the projected DOS should be `'atom X'` or `'orbital Y X'`, with 'X' being the chemical symbol and 'Y' the orbital label.
            These can be extracted from `entity_ref`.
        """,
    )

    def __init__(
        self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.name = self.m_def.name

    def resolve_energies_origin(
        self,
        energies: pint.Quantity,
        fermi_level: Optional[pint.Quantity],
        logger: 'BoundLogger',
    ) -> Optional[pint.Quantity]:
        """
        Resolve the origin of reference for the energies from the sibling `ElectronicEigenvalues` section and its
        `highest_occupied` level, or if this does not exist, from the `fermi_level` value as extracted from the sibling property, `FermiLevel`.

        Args:
            fermi_level (Optional[pint.Quantity]): The resolved Fermi level.
            energies (pint.Quantity): The grid points of the `Energy` variable.
            logger (BoundLogger): The logger to log messages.

        Returns:
            (Optional[pint.Quantity]): The resolved origin of reference for the energies.
        """
        # Check if the variables contain more than one variable (different than Energy)
        # ? Is this correct or should be use the index of energies to extract the proper shape element in `self.value` being used for `dos_values`?
        if len(self.variables) > 1:
            logger.warning(
                'The ElectronicDensityOfStates section contains more than one variable. We cannot extract the energy reference.'
            )
            return None

        # Extract the `ElectronicEigenvalues` section to get the `highest_occupied` and `lowest_unoccupied` energies
        # TODO implement once `ElectronicEigenvalues` is in the schema
        eigenvalues = get_sibling_section(
            section=self, sibling_section_name='electronic_eigenvalues', logger=logger
        )  # we consider `index_sibling` to be 0
        highest_occupied_energy = (
            eigenvalues.highest_occupied if eigenvalues is not None else None
        )
        lowest_unoccupied_energy = (
            eigenvalues.lowest_unoccupied if eigenvalues is not None else None
        )
        # and set defaults for `highest_occupied_energy` and `lowest_unoccupied_energy` in `m_cache`
        if highest_occupied_energy is not None:
            self.m_cache['highest_occupied_energy'] = highest_occupied_energy
        if lowest_unoccupied_energy is not None:
            self.m_cache['lowest_unoccupied_energy'] = lowest_unoccupied_energy

        # Check that the closest `energies` to the energy reference is not too far away.
        # If it is very far away, normalization may be very inaccurate and we do not report it.
        dos_values = self.value.magnitude
        eref = highest_occupied_energy if fermi_level is None else fermi_level
        fermi_idx = (np.abs(energies - eref)).argmin()
        fermi_energy_closest = energies[fermi_idx]
        distance = np.abs(fermi_energy_closest - eref)
        single_peak_fermi = False
        if distance.magnitude <= configuration.dos_energy_tolerance:
            # See if there are zero values close below the energy reference.
            idx = fermi_idx
            idx_descend = fermi_idx
            while True:
                try:
                    value = dos_values[idx]
                    energy_distance = np.abs(eref - energies[idx])
                except IndexError:
                    break
                if energy_distance.magnitude > configuration.dos_energy_tolerance:
                    break
                if value <= configuration.dos_intensities_threshold:
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
                if energy_distance.magnitude > configuration.dos_energy_tolerance:
                    break
                if value <= configuration.dos_intensities_threshold:
                    idx_ascend = idx
                    break
                idx += 1

            # If there is a single peak at fermi energy, no
            # search needs to be performed.
            if idx_ascend != fermi_idx and idx_descend != fermi_idx:
                self.m_cache['highest_occupied_energy'] = fermi_energy_closest
                self.m_cache['lowest_unoccupied_energy'] = fermi_energy_closest
                single_peak_fermi = True

            if not single_peak_fermi:
                # Look for highest occupied energy below the descend index
                idx = idx_descend
                while True:
                    try:
                        value = dos_values[idx]
                    except IndexError:
                        break
                    if value > configuration.dos_intensities_threshold:
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
                    if value > configuration.dos_intensities_threshold:
                        idx = idx if idx == idx_ascend else idx - 1
                        self.m_cache['highest_occupied_energy'] = energies[idx]
                        break
                    idx += 1

        # Return the `highest_occupied_energy` as the `energies_origin`, or the `fermi_level` if it is not None
        energies_origin = self.m_cache.get('highest_occupied_energy')
        if energies_origin is None:
            energies_origin = fermi_level
        return energies_origin

    def resolve_normalization_factor(self, logger: 'BoundLogger') -> Optional[float]:
        """
        Resolve the `normalization_factor` for the electronic DOS to get a cell-independent intensive DOS.

        Args:
            logger (BoundLogger): The logger to log messages.

        Returns:
            (Optional[float]): The normalization factor.
        """
        # Get the `ModelSystem` as referenced in the `Outputs.model_system_ref`
        model_system = get_sibling_section(
            section=self, sibling_section_name='model_system_ref', logger=logger
        )
        if model_system is None:
            logger.warning(
                'Could not resolve the referenced `ModelSystem` in the `Outputs`.'
            )
            return None

        # Get the originally parsed `AtomicCell`, which is the first element stored in `ModelSystem.cell` of name `'AtomicCell'`
        atomic_cell = None
        for cell in model_system.cell:
            if cell.name == 'AtomicCell':  # we get the originally parsed `AtomicCell`
                atomic_cell = cell
                break
        if atomic_cell is None:
            logger.warning(
                'Could not resolve the `AtomicCell` from the referenced `ModelSystem`.'
            )
            return None

        # Get the `atoms_state` and their `atomic_number` from the `AtomicCell`
        if atomic_cell.atoms_state is None or len(atomic_cell.atoms_state) == 0:
            logger.warning('Could not resolve the `atoms_state` from the `AtomicCell`.')
            return None
        atomic_numbers = [atom.atomic_number for atom in atomic_cell.atoms_state]

        # Return `normalization_factor` depending if the calculation is spin polarized or not
        if self.spin_channel is not None:
            normalization_factor = 1 / (2 * sum(atomic_numbers))
        else:
            normalization_factor = 1 / sum(atomic_numbers)
        return normalization_factor

    def extract_band_gap(self) -> Optional[ElectronicBandGap]:
        """
        Extract the electronic band gap from the `highest_occupied_energy` and `lowest_unoccupied_energy` stored
        in `m_cache` from `resolve_energies_origin()`. If the difference of `highest_occupied_energy` and
        `lowest_unoccupied_energy` is negative, the band gap `value` is set to 0.0.

        Returns:
            (Optional[ElectronicBandGap]): The extracted electronic band gap section to be stored in `Outputs`.
        """
        band_gap = None
        homo = self.m_cache.get('highest_occupied_energy')
        lumo = self.m_cache.get('lowest_unoccupied_energy')
        if homo and lumo:
            band_gap = ElectronicBandGap()
            band_gap.is_derived = True
            band_gap.physical_property_ref = self

            if (homo - lumo).magnitude < 0:
                band_gap.value = 0.0
            else:
                band_gap.value = homo - lumo
        return band_gap

    def extract_projected_dos(
        self, type: str, logger: 'BoundLogger'
    ) -> list[Optional[DOSProfile]]:
        """
        Extract the projected DOS from the `projected_dos` section and the specified `type`.

        Args:
            type (str): The type of the projected DOS to extract. It can be `'atom'` or `'orbital'`.

        Returns:
            (DOSProfile): The extracted projected DOS.
        """
        extracted_pdos = []
        for pdos in self.projected_dos:
            # We make sure each PDOS is normalized
            pdos.normalize(None, logger)

            # Initial check for `name` and `entity_ref`
            if (
                pdos.name is None
                or pdos.entity_ref is None
                or len(pdos.entity_ref) == 0
            ):
                logger.warning(
                    '`name` or `entity_ref` are not set for `projected_dos` and they are required for normalization to work.'
                )
                return None

            if type in pdos.name:
                extracted_pdos.append(pdos)
        return extracted_pdos

    def generate_from_projected_dos(
        self, logger: 'BoundLogger'
    ) -> Optional[pint.Quantity]:
        """
        Generate the total `value` of the electronic DOS from the `projected_dos` contributions. If the `projected_dos`
        is not present, it returns `None`.

        Args:
            logger (BoundLogger): The logger to log messages.

        Returns:
            (Optional[pint.Quantity]): The total `value` of the electronic DOS.
        """
        if self.projected_dos is None or len(self.projected_dos) == 0:
            return None

        # Extract `Energy` variables
        energies = get_variables(self.variables, Energy)
        if len(energies) != 1:
            logger.warning(
                'The `ElectronicDensityOfStates` does not contain an `Energy` variable to extract the DOS.'
            )
            return None

        # We distinguish between orbital and atom `projected_dos`
        orbital_projected = self.extract_projected_dos('orbital', logger)
        atom_projected = self.extract_projected_dos('atom', logger)

        # Extract `atom_projected` from `orbital_projected` by summing up the `orbital_projected` contributions for each atom
        if len(atom_projected) == 0:
            atom_data: dict[AtomsState, list[DOSProfile]] = {}
            for orb_pdos in orbital_projected:
                # `entity_ref` is the `OrbitalsState` section, whose parent is `AtomsState`
                entity_ref = orb_pdos.entity_ref.m_parent
                if entity_ref in atom_data:
                    atom_data[entity_ref].append(orb_pdos)
                else:
                    atom_data[entity_ref] = [orb_pdos]
            for ref, data in atom_data.items():
                atom_dos = DOSProfile(
                    name=f'atom {ref.chemical_symbol}',
                    entity_ref=ref,
                    variables=energies,
                )
                orbital_values = [
                    dos.value.magnitude for dos in data
                ]  # to avoid warnings from pint
                orbital_unit = data[0].value.u
                atom_dos.value = np.sum(orbital_values, axis=0) * orbital_unit
                atom_projected.append(atom_dos)
            # We concatenate the `atom_projected` to the `projected_dos`
            self.projected_dos = orbital_projected + atom_projected

        # Extract `value` from `atom_projected` by summing up the `atom_projected` contributions
        value = self.value
        if value is None:
            atom_values = [
                dos.value.magnitude for dos in atom_projected
            ]  # to avoid warnings from pint
            atom_unit = atom_projected[0].value.u
            value = np.sum(atom_values, axis=0) * atom_unit
        return value

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

        # Initial check to see if `variables` contains the required `Energy` variable
        energies = get_variables(self.variables, Energy)
        if len(energies) != 0:
            return
        energies = energies[0].points

        # Resolve `fermi_level` from a sibling section with respect to `ElectronicDensityOfStates`
        fermi_level = get_sibling_section(
            section=self, sibling_section_name='fermi_level', logger=logger
        )  # * we consider `index_sibling` to be 0
        if fermi_level is not None:
            fermi_level = fermi_level.value
        # and the `energies_origin` from the sibling `ElectronicEigenvalues` section
        self.energies_origin = self.resolve_energies_origin(
            energies, fermi_level, logger
        )
        if self.energies_origin is None:
            logger.info('Could not resolve the `energies_origin` for the DOS')

        # Resolve `normalization_factor`
        if self.normalization_factor is None:
            self.normalization_factor = self.resolve_normalization_factor(logger)

        # `ElectronicBandGap` extraction
        band_gap = self.extract_band_gap()
        if band_gap is not None:
            self.m_parent.electronic_band_gap.append(band_gap)

        # Total `value` extraction from `projected_dos`
        value_from_pdos = self.generate_from_projected_dos(logger)
        if self.value is None and value_from_pdos is not None:
            logger.info(
                'The `ElectronicDensityOfStates.value` is not stored. We will attempt to obtain it by summing up projected DOS contributions, if these are present.'
            )
            self.value = value_from_pdos


class AbsorptionSpectrum(SpectralProfile):
    """ """

    # ! implement `iri` and `rank` as part of `m_def = Section()`

    axis = Quantity(
        type=MEnum('xx', 'yy', 'zz'),
        description="""
        Axis of the absorption spectrum. This is related with the polarization direction, and can be seen as the
        principal term in the tensor `Permittivity.value` (see permittivity.py module).
        """,
    )

    def __init__(
        self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        # Set the name of the section
        self.name = self.m_def.name

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class XASSpectrum(AbsorptionSpectrum):
    """
    X-ray Absorption Spectrum (XAS).
    """

    # ! implement `iri` and `rank` as part of `m_def = Section()`

    xanes_spectrum = SubSection(
        sub_section=AbsorptionSpectrum.m_def,
        description="""
        X-ray Absorption Near Edge Structure (XANES) spectrum.
        """,
        repeats=False,
    )

    exafs_spectrum = SubSection(
        sub_section=AbsorptionSpectrum.m_def,
        description="""
        Extended X-ray Absorption Fine Structure (EXAFS) spectrum.
        """,
        repeats=False,
    )

    def __init__(
        self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        # Set the name of the section
        self.name = self.m_def.name

    def generate_from_contributions(self, logger: 'BoundLogger') -> None:
        """
        Generate the `value` of the XAS spectrum by concatenating the XANES and EXAFS contributions. It also concatenates
        the `Energy` grid points of the XANES and EXAFS parts.

        Args:
            logger (BoundLogger): The logger to log messages.
        """
        # TODO check if this method is general enough
        if self.xanes_spectrum is not None and self.exafs_spectrum is not None:
            # Concatenate XANE and EXAFS `Energy` grid points
            xanes_variables = get_variables(self.xanes_spectrum.variables, Energy)
            exafs_variables = get_variables(self.exafs_spectrum.variables, Energy)
            if len(xanes_variables) == 0 or len(exafs_variables) == 0:
                logger.warning(
                    'Could not extract the `Energy` grid points from XANES or EXAFS.'
                )
                return
            xanes_energies = xanes_variables[0].points
            exafs_energies = exafs_variables[0].points
            if xanes_energies.max() > exafs_energies.min():
                logger.warning(
                    'The XANES `Energy` grid points are not below the EXAFS `Energy` grid points.'
                )
                return
            self.variables = [
                Energy(points=np.concatenate([xanes_energies, exafs_energies]))
            ]
            # Concatenate XANES and EXAFS `value` if they have the same shape ['n_energies']
            try:
                self.value = np.concatenate(
                    [self.xanes_spectrum.value, self.exafs_spectrum.value]
                )
            except ValueError:
                logger.warning(
                    'The XANES and EXAFS `value` have different shapes. Could not concatenate the values.'
                )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

        if self.value is None:
            logger.info(
                'The `XASSpectrum.value` is not stored. We will attempt to obtain it by combining the XANES and EXAFS parts if these are present.'
            )
            self.generate_from_contributions(logger)
