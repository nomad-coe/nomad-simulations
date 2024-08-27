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

from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import pint
from nomad.config import config
from nomad.metainfo import Quantity, SubSection

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from nomad.metainfo import Context, Section
    from structlog.stdlib import BoundLogger

from nomad_simulations.schema_packages.atoms_state import AtomsState, OrbitalsState
from nomad_simulations.schema_packages.numerical_settings import KSpace
from nomad_simulations.schema_packages.physical_property import (
    PhysicalProperty,
    validate_quantity_wrt_value,
)
from nomad_simulations.schema_packages.properties.band_gap import ElectronicBandGap
from nomad_simulations.schema_packages.properties.fermi_surface import FermiSurface
from nomad_simulations.schema_packages.utils import get_sibling_section

configuration = config.get_plugin_entry_point(
    'nomad_simulations.schema_packages:nomad_simulations_plugin'
)


class BaseElectronicEigenvalues(PhysicalProperty):
    """
    A base section used to define basic quantities for the `ElectronicEigenvalues`  and `ElectronicBandStructure` properties.
    """

    iri = ''

    n_bands = Quantity(
        type=np.int32,
        description="""
        Number of bands / eigenvalues.
        """,
    )

    value = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        Value of the electronic eigenvalues.
        """,
    )

    def __init__(
        self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        # ! `n_bands` need to be set up during initialization of the class
        self.rank = [int(kwargs.get('n_bands'))]

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class ElectronicEigenvalues(BaseElectronicEigenvalues):
    """ """

    iri = 'http://fairmat-nfdi.eu/taxonomy/ElectronicEigenvalues'

    spin_channel = Quantity(
        type=np.int32,
        description="""
        Spin channel of the corresponding electronic eigenvalues. It can take values of 0 or 1.
        """,
    )

    occupation = Quantity(
        type=np.float64,
        shape=['*', 'n_bands'],
        description="""
        Occupation of the electronic eigenvalues. This is a number depending whether the `spin_channel` has been set or not.
        If `spin_channel` is set, then this number is between 0 and 1, where 0 means that the state is unoccupied and 1 means
        that the state is fully occupied; if `spin_channel` is not set, then this number is between 0 and 2. The shape of
        this quantity is defined as `[K.n_points, K.dimensionality, n_bands]`, where `K` is a `variable` which can
        be `KMesh` or `KLinePath`, depending whether the simulation mapped the whole Brillouin zone or just a specific
        path.
        """,
    )

    highest_occupied = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        Highest occupied electronic eigenvalue. Together with `lowest_unoccupied`, it defines the
        electronic band gap.
        """,
    )

    lowest_unoccupied = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        Lowest unoccupied electronic eigenvalue. Together with `highest_occupied`, it defines the
        electronic band gap.
        """,
    )

    # ? Should we add functionalities to handle min/max of the `value` in some specific cases, .e.g, bands around the Fermi level,
    # ? core bands separated by gaps, and equivalently, higher-energy valence bands separated by gaps?

    value_contributions = SubSection(
        sub_section=BaseElectronicEigenvalues.m_def,
        repeats=True,
        description="""
        Contributions to the electronic eigenvalues. Example, in the case of a DFT+GW calculation, the GW eigenvalues
        are stored under `value`, and each contribution is identified by `label`:
            - `'KS'`: Kohn-Sham contribution. This is also stored in the DFT entry under `ElectronicEigenvalues.value`.
            - `'KSxc'`: Diagonal matrix elements of the expectation value of the Kohn-Sahm exchange-correlation potential.
            - `'SigX'`: Diagonal matrix elements of the exchange self-energy. This is also stored in the GW entry under `ElectronicSelfEnergy.value`.
            - `'SigC'`: Diagonal matrix elements of the correlation self-energy. This is also stored in the GW entry under `ElectronicSelfEnergy.value`.
            - `'Zk'`: Quasiparticle renormalization factors contribution. This is also stored in the GW entry under `QuasiparticleWeights.value`.
        """,
    )

    reciprocal_cell = Quantity(
        type=KSpace.reciprocal_lattice_vectors,
        description="""
        Reference to the reciprocal lattice vectors stored under `KSpace`.
        """,
    )

    def __init__(
        self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.name = self.m_def.name

    @validate_quantity_wrt_value(name='occupation')
    def order_eigenvalues(self) -> Union[bool, tuple[pint.Quantity, np.ndarray]]:
        """
        Order the eigenvalues based on the `value` and `occupation`. The return `value` and
        `occupation` are flattened.

        Returns:
            (Union[bool, tuple[pint.Quantity, np.ndarray]]): The flattened and sorted `value` and `occupation`. If validation
            fails, then it returns `False`.
        """
        total_shape = np.prod(self.value.shape)

        # Order the indices in the flattened list of `value`
        flattened_value = self.value.reshape(total_shape)
        flattened_occupation = self.occupation.reshape(total_shape)
        sorted_indices = np.argsort(flattened_value, axis=0)

        sorted_value = (
            np.take_along_axis(flattened_value.magnitude, sorted_indices, axis=0)
            * flattened_value.u
        )
        sorted_occupation = np.take_along_axis(
            flattened_occupation, sorted_indices, axis=0
        )
        self.m_cache['sorted_eigenvalues'] = True
        return sorted_value, sorted_occupation

    def resolve_homo_lumo_eigenvalues(
        self,
    ) -> tuple[Optional[pint.Quantity], Optional[pint.Quantity]]:
        """
        Resolve the `highest_occupied` and `lowest_unoccupied` eigenvalues by performing a binary search on the
        flattened and sorted `value` and `occupation`. If these quantities already exist, overwrite them or return
        them if it is not possible to resolve from `value` and `occupation`.

        Returns:
            (tuple[Optional[pint.Quantity], Optional[pint.Quantity]]): The `highest_occupied` and
            `lowest_unoccupied` eigenvalues.
        """
        # Sorting `value` and `occupation`
        if not self.order_eigenvalues():  # validation fails
            if self.highest_occupied is not None and self.lowest_unoccupied is not None:
                return self.highest_occupied, self.lowest_unoccupied
            return None, None
        sorted_value, sorted_occupation = self.order_eigenvalues()
        sorted_value_unit = sorted_value.u
        sorted_value = sorted_value.magnitude

        # Binary search ot find the transition point between `occupation = 2` and `occupation = 0`
        homo = self.highest_occupied
        lumo = self.lowest_unoccupied
        mid = (
            np.searchsorted(
                sorted_occupation <= configuration.occupation_tolerance, True
            )
            - 1
        )
        if mid >= 0 and mid < len(sorted_occupation) - 1:
            if sorted_occupation[mid] > 0 and (
                sorted_occupation[mid + 1] >= -configuration.occupation_tolerance
                and sorted_occupation[mid + 1] <= configuration.occupation_tolerance
            ):
                homo = sorted_value[mid] * sorted_value_unit
                lumo = sorted_value[mid + 1] * sorted_value_unit

        return homo, lumo

    def extract_band_gap(self) -> Optional[ElectronicBandGap]:
        """
        Extract the electronic band gap from the `highest_occupied` and `lowest_unoccupied` eigenvalues.
        If the difference of `highest_occupied` and `lowest_unoccupied` is negative, the band gap `value` is set to 0.0.

        Returns:
            (Optional[ElectronicBandGap]): The extracted electronic band gap section to be stored in `Outputs`.
        """
        band_gap = None
        homo, lumo = self.resolve_homo_lumo_eigenvalues()
        if homo and lumo:
            band_gap = ElectronicBandGap(is_derived=True, physical_property_ref=self)

            if (lumo - homo).magnitude < 0:
                band_gap.value = 0.0
            else:
                band_gap.value = lumo - homo
        return band_gap

    # TODO fix this method once `FermiSurface` property is implemented
    def extract_fermi_surface(self, logger: 'BoundLogger') -> Optional[FermiSurface]:
        """
        Extract the Fermi surface for metal systems and using the `FermiLevel.value`.

        Args:
            logger (BoundLogger): The logger to log messages.

        Returns:
            (Optional[FermiSurface]): The extracted Fermi surface section to be stored in `Outputs`.
        """
        # Check if the system has a finite band gap
        homo, lumo = self.resolve_homo_lumo_eigenvalues()
        if (homo and lumo) and (lumo - homo).magnitude > 0:
            return None

        # Get the `fermi_level.value`
        fermi_level = get_sibling_section(
            section=self, sibling_section_name='fermi_level', logger=logger
        )
        if fermi_level is None:
            logger.warning(
                'Could not extract the `FermiSurface`, because `FermiLevel` is not stored.'
            )
            return None
        fermi_level_value = fermi_level.value.magnitude

        # Extract values close to the `fermi_level.value`
        fermi_indices = np.logical_and(
            self.value.magnitude
            >= (fermi_level_value - configuration.fermi_surface_tolerance),
            self.value.magnitude
            <= (fermi_level_value + configuration.fermi_surface_tolerance),
        )
        fermi_values = self.value[fermi_indices]

        # Store `FermiSurface` values
        # ! This is wrong (!) the `value` should be the `KMesh.points`, not the `ElectronicEigenvalues.value`
        fermi_surface = FermiSurface(
            n_bands=self.n_bands,
            is_derived=True,
            physical_property_ref=self,
        )
        fermi_surface.variables = self.variables
        fermi_surface.value = fermi_values
        return fermi_surface

    def resolve_reciprocal_cell(self) -> Optional[pint.Quantity]:
        """
        Resolve the reciprocal cell from the `KSpace` numerical settings section.

        Returns:
            Optional[pint.Quantity]: _description_
        """
        numerical_settings = self.m_xpath(
            'm_parent.m_parent.model_method[-1].numerical_settings', dict=False
        )
        if numerical_settings is None:
            return None
        k_space = None
        for setting in numerical_settings:
            if isinstance(setting, KSpace):
                k_space = setting
                break
        if k_space is None:
            return None
        return k_space

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

        # Resolve `highest_occupied` and `lowest_unoccupied` eigenvalues
        self.highest_occupied, self.lowest_unoccupied = (
            self.resolve_homo_lumo_eigenvalues()
        )

        # `ElectronicBandGap` extraction
        band_gap = self.extract_band_gap()
        if band_gap is not None:
            self.m_parent.electronic_band_gaps.append(band_gap)

        # TODO uncomment once `FermiSurface` property is implemented
        # `FermiSurface` extraction
        # fermi_surface = self.extract_fermi_surface(logger)
        # if fermi_surface is not None:
        #     self.m_parent.fermi_surfaces.append(fermi_surface)

        # Resolve `reciprocal_cell` from the `KSpace` numerical settings section
        self.reciprocal_cell = self.resolve_reciprocal_cell()


class ElectronicBandStructure(ElectronicEigenvalues):
    """
    Accessible energies by the charges (electrons and holes) in the reciprocal space.
    """

    iri = 'http://fairmat-nfdi.eu/taxonomy/ElectronicBandStructure'

    def __init__(
        self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.name = self.m_def.name

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class Occupancy(PhysicalProperty):
    """
    Electrons occupancy of an atom per orbital and spin. This is a number defined between 0 and 1 for
    spin-polarized systems, and between 0 and 2 for non-spin-polarized systems. This property is
    important when studying if an orbital or spin channel are fully occupied, at half-filling, or
    fully emptied, which have an effect on the electron-electron interaction effects.
    """

    iri = 'http://fairmat-nfdi.eu/taxonomy/Occupancy'

    atoms_state_ref = Quantity(
        type=AtomsState,
        description="""
        Reference to the `AtomsState` section in which the occupancy is calculated.
        """,
    )

    orbitals_state_ref = Quantity(
        type=OrbitalsState,
        description="""
        Reference to the `OrbitalsState` section in which the occupancy is calculated.
        """,
    )

    spin_channel = Quantity(
        type=np.int32,
        description="""
        Spin channel of the corresponding electronic property. It can take values of 0 and 1.
        """,
    )

    value = Quantity(
        type=np.float64,
        description="""
        Value of the electronic occupancy in the atom defined by `atoms_state_ref` and the orbital
        defined by `orbitals_state_ref`. the orbital. If `spin_channel` is set, then this number is
        between 0 and 1, where 0 means that the state is unoccupied and 1 means that the state is
        fully occupied; if `spin_channel` is not set, then this number is between 0 and 2.
        """,
    )

    def __init__(
        self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.name = self.m_def.name

    # TODO add extraction from `ElectronicEigenvalues.occupation`

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)
