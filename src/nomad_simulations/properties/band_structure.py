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
from typing import Optional, Tuple
import pint

from nomad.metainfo import Quantity, Section, Context, SubSection

from nomad_simulations.numerical_settings import KSpace
from nomad_simulations.physical_property import PhysicalProperty
from nomad_simulations.properties import ElectronicBandGap, FermiSurface
from nomad_simulations.utils import get_sibling_section


class BaseElectronicEigenvalues(PhysicalProperty):
    """
    A base section used to define basic quantities for the `ElectronicEigenvalues`, `FermiSurface`, and
    `ElectronicBandStructure` properties. This section serves to define `FermiSurface` and without needing to specify
    other quantities that appear in `ElectronicEigenvalues`
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
        self, m_def: Section = None, m_context: Context = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        # ! `n_bands` need to be set up during initialization of the class
        self.rank = [self.n_bands]

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class ElectronicEigenvalues(BaseElectronicEigenvalues):
    """ """

    iri = 'http://fairmat-nfdi.eu/taxonomy/ElectronicEigenvalues'

    occupation = Quantity(
        type=np.float64,
        shape=['*', 'n_bands'],
        description="""
        Occupation of the electronic eigenvalues. This is a number between 0 and 2, where 0 means
        that the state is unoccupied and 2 means that the state is fully occupied. It is controlled
        by the Fermi-Dirac distribution:

            $ f(E) = 1 / (1 + exp((E - E_F) / kT)) $

        The shape of this quantity is defined as `[KMesh.n_points, KMesh.dimensionality, n_bands]`, where `KMesh` is a `variable`.
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
        Reference to the reciprocal lattice vectors stored under `KSpace`. This reference is useful when resolving the Brillouin zone
        for the front-end visualization.
        """,
    )

    def __init__(
        self, m_def: Section = None, m_context: Context = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.name = self.m_def.name

    def _check_occupation_shape(self, logger: BoundLogger) -> bool:
        """
        Check if `occupation` exists and have the same shape as `value`.

        Args:
            logger (BoundLogger): The logger to log messages.

        Returns:
            (bool): True if the shape of `occupation` is the same as `value`, False otherwise.
        """
        if self.occupation is None or len(self.occupation) == 0:
            logger.warning('Cannot find `occupation` defined.')
            return False
        if self.value is not None and self.value.shape != self.occupation.shape:
            logger.warning(
                'The shape of `value` and `occupation` are different. They should have the same shape.'
            )
            return False
        return True

    def order_eigenvalues(
        self, logger: BoundLogger
    ) -> Tuple[Optional[pint.Quantity], Optional[np.ndarray]]:
        """
        Order the eigenvalues based on the `value` and `occupation`. The return `value` and
        `occupation` are flattened.

        Args:
            logger (BoundLogger): The logger to log messages.

        Returns:
            (Tuple[Optional[pint.Quantity], Optional[list]]): The flattened and sorted `value` and `occupation`.
        """

        # Check if `value` exists
        if self.value is None or len(self.value) == 0:
            logger.error('Could not find `value` defined.')
            return None, None

        # Check if `occupation` exists and have the same shape as `value`
        if not self._check_occupation_shape(logger):
            return None, None
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
        self, logger: BoundLogger
    ) -> Tuple[Optional[pint.Quantity], Optional[pint.Quantity]]:
        """
        Resolve the `highest_occupied` and `lowest_unoccupied` eigenvalues by performing a binary search on the
        flattened and sorted `value` and `occupation`. If these quantities already exist, overwrite them or return
        them if it is not possible to resolve from `value` and `occupation`.

        Args:
            logger (BoundLogger): The logger to log messages.

        Returns:
            (Tuple[Optional[pint.Quantity], Optional[pint.Quantity]]): The `highest_occupied` and
            `lowest_unoccupied` eigenvalues.
        """
        # Sorting `value` and `occupation`
        sorted_value, sorted_occupation = self.order_eigenvalues(logger)
        if sorted_value is None or sorted_occupation is None:
            if self.highest_occupied is not None and self.lowest_unoccupied is not None:
                return self.highest_occupied, self.lowest_unoccupied
            return None, None
        sorted_value_unit = sorted_value.u
        sorted_value = sorted_value.magnitude

        # Binary search ot find the transition point between `occupation = 2` and `occupation = 0`
        tolerance = 1e-3  # TODO add tolerance from config fields
        homo = self.highest_occupied
        lumo = self.lowest_unoccupied
        low_occ = 0
        high_unocc = sorted_value.shape[-1] - 1
        while low_occ <= high_unocc:
            mid = (low_occ + high_unocc) // 2
            # check if occupation[mid] and [mid+1] is 0
            if sorted_occupation[mid] > 0 and (
                sorted_occupation[mid + 1] >= -tolerance
                and sorted_occupation[mid + 1] <= tolerance
            ):
                homo = sorted_value[mid] * sorted_value_unit
                lumo = sorted_value[mid + 1] * sorted_value_unit
                break
            # check if the occupation[mid] is finite but [mid+1] is as well
            elif sorted_occupation[mid] > 0:
                low_occ = mid + 1
            # check if the occupation[mid] is 0
            else:
                high_unocc = mid - 1

        return homo, lumo

    def extract_band_gap(self, logger: BoundLogger) -> Optional[ElectronicBandGap]:
        """
        Extract the electronic band gap from the `highest_occupied` and `lowest_unoccupied` eigenvalues.
        If the difference of `highest_occupied` and `lowest_unoccupied` is negative, the band gap `value` is set to 0.0.

        Args:
            logger (BoundLogger): The logger to log messages.

        Returns:
            (Optional[ElectronicBandGap]): The extracted electronic band gap section to be stored in `Outputs`.
        """
        band_gap = None
        homo, lumo = self.resolve_homo_lumo_eigenvalues(logger)
        if homo and lumo:
            band_gap = ElectronicBandGap(is_derived=True, physical_property_ref=self)

            if (lumo - homo).magnitude < 0:
                band_gap.value = 0.0
            else:
                band_gap.value = lumo - homo
        return band_gap

    # TODO fix this method once `FermiSurface` property is implemented
    def extract_fermi_surface(self, logger: BoundLogger) -> Optional[FermiSurface]:
        """
        Extract the Fermi surface for metal systems and using the `FermiLevel.value`.

        Args:
            logger (BoundLogger): The logger to log messages.

        Returns:
            (Optional[FermiSurface]): The extracted Fermi surface section to be stored in `Outputs`.
        """
        # Check if the system has a finite band gap
        homo, lumo = self.resolve_homo_lumo_eigenvalues(logger)
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
        tolerance = 1e-8  # TODO change this for a config field
        fermi_indices = np.logical_and(
            self.value.magnitude >= (fermi_level_value - tolerance),
            self.value.magnitude <= (fermi_level_value + tolerance),
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

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        # Check if `occupation` exists and have the same shape as `value`
        if not self._check_occupation_shape(logger):
            return

        # Resolve `highest_occupied` and `lowest_unoccupied` eigenvalues
        self.highest_occupied, self.lowest_unoccupied = (
            self.resolve_homo_lumo_eigenvalues(logger)
        )

        # `ElectronicBandGap` extraction
        band_gap = self.extract_band_gap(logger)
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
        self, m_def: Section = None, m_context: Context = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.name = self.m_def.name

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)
