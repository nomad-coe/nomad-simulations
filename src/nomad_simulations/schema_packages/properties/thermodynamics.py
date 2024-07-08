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

from typing import TYPE_CHECKING

import numpy as np
from nomad.metainfo import Quantity

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from nomad.metainfo import Context, Section
    from structlog.stdlib import BoundLogger

from nomad_simulations.schema_packages.physical_property import PhysicalProperty
from nomad_simulations.schema_packages.properties.energies import BaseEnergy

######################################
# fundamental thermodynamic properties
######################################


class Pressure(PhysicalProperty):
    """
    The force exerted per unit area by gas particles as they collide with the walls of
    their container.
    """

    # iri = 'http://fairmat-nfdi.eu/taxonomy/Pressure' # ! Does not yet exist in taxonomy

    value = Quantity(
        type=np.float64,
        unit='pascal',
        description="""
        """,
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class Volume(PhysicalProperty):
    """
    the amount of three-dimensional space that a substance or material occupies.
    """

    #! Above description suggested for taxonomy
    # TODO check back on definition after first taxonomy version

    iri = 'http://fairmat-nfdi.eu/taxonomy/Volume'

    value = Quantity(
        type=np.float64,
        unit='m ** 3',
        description="""
        """,
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class Temperature(PhysicalProperty):
    """
    a measure of the average kinetic energy of the particles in a system.
    """

    value = Quantity(
        type=np.float64,
        unit='kelvin',
        description="""
        """,
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class Heat(BaseEnergy):
    """
    The transfer of thermal energy **into** a system.
    """

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class Work(BaseEnergy):
    """
    The energy transferred to a system by means of force applied over a distance.
    """

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class InternalEnergy(BaseEnergy):
    """
    The total energy contained within a system, encompassing both kinetic and potential
    energies of the particles. The change in `InternalEnergy` for some thermodynamic
    process may be expressed as the `Heat` minus the `Work`.
    """

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class Enthalpy(BaseEnergy):
    """
    The total heat content of a system, defined as 'InternalEnergy' + 'Pressure' * 'Volume'.
    """

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class Entropy(PhysicalProperty):
    """
    A measure of the disorder or randomness in a system.

    From a thermodynamic perspective, `Entropy` is a measure of the system's energy
    dispersal at a specific temperature, and can be interpreted as the unavailability of
    a system's thermal energy for conversion into mechanical work. For a reversible
    process, the change in `Entropy` is given mathematically by an integral over the
    infinitesimal `Heat` (i.e., thermal energy transfered into the system) divided by the
    `Temperature`.

    From a statistical mechanics viewpoint, entropy quantifies the number of microscopic
    configurations (microstates) that correspond to a thermodynamic system's macroscopic
    state, as given by the Boltzmann equation for entropy.
    """

    value = Quantity(
        type=np.float64,
        unit='joule / kelvin',
        description="""
        """,
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class GibbsFreeEnergy(BaseEnergy):
    """
    The energy available to do work in a system at constant temperature and pressure,
    given by `Enthalpy` - `Temperature` * `Entropy`.
    """

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class HelmholtzFreeEnergy(BaseEnergy):
    """
    The energy available to do work in a system at constant volume and temperature,
    given by `InternalEnergy` - `Temperature` * `Entropy`.
    """

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class ChemicalPotential(BaseEnergy):
    """
    Free energy cost of adding or extracting a particle from a thermodynamic system.
    """

    # ! implement `iri` and `rank` as part of `m_def = Section()`

    iri = 'http://fairmat-nfdi.eu/taxonomy/ChemicalPotential'

    def __init__(
        self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.rank = []
        self.name = self.m_def.name

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class HeatCapacity(PhysicalProperty):
    """
    Amount of heat to be supplied to a material to produce a unit change in its temperature.
    """

    value = Quantity(
        type=np.float64,
        unit='joule / kelvin',
        description="""
        """,
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


################################
# other thermodynamic properties
################################


class VirialTensor(BaseEnergy):
    """
    A measure of the distribution of internal forces and the overall stress within
    a system of particles. Mathematically, the virial tensor is defined as minus the sum
    of the dot product between the position and force vectors for each particle.
    The `VirialTensor` can be related to the non-ideal pressure of the system through
    the virial theorem.
    """

    def __init__(
        self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.rank = [3, 3]
        self.name = self.m_def.name

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class MassDensity(PhysicalProperty):
    """
    Mass per unit volume of a material.
    """

    value = Quantity(
        type=np.float64,
        unit='kg / m ** 3',
        description="""
        """,
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


# ? fit better elsewhere
class Hessian(PhysicalProperty):
    """
    A square matrix of second-order partial derivatives of a potential energy function,
    describing the local curvature of the energy surface.
    """

    value = Quantity(
        type=np.float64,
        unit='joule / m ** 2',
        description="""
        """,
    )

    def __init__(
        self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.rank = [3, 3]
        self.name = self.m_def.name

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)
