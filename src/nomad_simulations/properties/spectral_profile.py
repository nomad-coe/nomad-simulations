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

from nomad.units import ureg
from nomad.metainfo import (
    Quantity,
    SubSection,
    MEnum,
    Section,
    Context,
    JSON,
)

from ..physical_property import PhysicalProperty


class SpectralProfile(PhysicalProperty):
    """
    A base section used to define the spectral profile.
    """

    value = Quantity(
        type=np.float64,
        description="""
        Value of the intensities of a spectral profile in arbitrary units.
        """,
    )

    # TODO implement this in PhysicalProperty (similar to shape)
    # value_units = Quantity(
    #     type=str,
    #     description="""
    #     Unit using the pint UnitRegistry() notation for the `value`. Example, if the spectra `is_derived`
    #     from the imaginary part of the dielectric function, `value` are `'F/m'`.
    #     """,
    # )

    def __init__(self, m_def: Section = None, m_context: Context = None, **kwargs):
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
    Electronic Density of States (DOS).
    """

    iri = 'http://fairmat-nfdi.eu/taxonomy/ElectronicDensityOfStates'

    spin_channel = Quantity(
        type=np.int32,
        description="""
        Spin channel of the corresponding DOS. It can take values of 0 or 1. This quantity is set only if
        `ModelMethod.is_spin_polarized` is `True`.
        """,
    )

    fermi_level = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        The Fermi level is the highest occupied energy level at zero temperature. For insulators and semiconductors,
        some codes have some difficulty in determining the Fermi level, and it is often set to the middle of the band gap,
        at the top of the valence band, or even in the bottom of the conduction band.

        We set the `energies_origin` to the top of the valence band, thus, `energies_origin` and `fermi_level` might
        not coincide.
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

    def __init__(self, m_def: Section = None, m_context: Context = None, **kwargs):
        super().__init__(m_def, m_context, **kwargs)
        self.name = self.m_def.name

    def resolve_fermi_level(self) -> Optional[np.float64]:
        """
        Resolve the Fermi level from the output `FermiLevel` section, if present.

        Returns:
            (Optional[np.float64]): The resolved Fermi level.
        """
        fermi_level = self.fermi_level
        # if fermi_level is None:
        #     fermi_level = resolve_output_value(self, FermiLevel)
        # return fermi_level

    def check_spin_polarized(self) -> bool:
        """
        Check if the simulation `is_spin_polarized`.

        Returns:
            (bool): True if the simulation is spin-polarized, False otherwise.
        """
        if self.m_parent is not None:
            for method in self.m_parent.model_method:
                if method.is_spin_polarized:
                    return True
        return False

    def resolve_energies_origin(self) -> Optional[pint.Quantity]:
        """
        Resolve the origin of reference for the energies from the `Eigenvalues` output property, or if this does not exist,
        from the `FermiLevel` output property.

        Returns:
            (Optional[pint.Quantity]): The resolved origin of reference for the energies.
        """
        energies_origin = self.energies_origin
        # ! We need schema for `Eigenvalues` to store `highest_occupied` and `lowest_occupied` to use in `ElectronicBandGap` and `ElectronicDOS`
        # if energies_origin is None:
        #     fermi_level = self.resolve_fermi_level()
        #     if fermi_level is not None:
        #         energies_origin = fermi_level
        #     else:
        #         eigenvalues = resolve_output_value(self, Eigenvalues)
        #         if eigenvalues is not None:
        #             energies_origin = eigenvalues.highest_occupied
        return energies_origin

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        # Resolve `fermi_level`
        self.fermi_level = self.resolve_fermi_level()

        # Check if `is_spin_polarized` but `spin_channel` is not set
        if self.check_spin_polarized() and self.spin_channel is None:
            logger.warning(
                'Spin-polarized calculation detected but the `spin_channel` is not set.'
            )

        # Resolve `energies_origin`
        self.energies_origin = self.resolve_energies_origin()


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

    def __init__(self, m_def: Section = None, m_context: Context = None, **kwargs):
        super().__init__(m_def, m_context, **kwargs)
        # Set the name of the section
        self.name = self.m_def.name

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)
