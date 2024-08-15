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
from nomad.metainfo import Context, Quantity, Section, SubSection

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from nomad.metainfo import Context, Section
    from structlog.stdlib import BoundLogger

from nomad_simulations.schema_packages.physical_property import (
    PhysicalProperty,
    PropertyContribution,
)

##################
# Abstract classes
##################


class BaseEnergy(PhysicalProperty):
    """
    Abstract class used to define a common `value` quantity with the appropriate units
    for different types of energies, which avoids repeating the definitions for each
    energy class.
    """

    value = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        """,
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class EnergyContribution(BaseEnergy, PropertyContribution):
    """
    Abstract class for incorporating specific energy contributions to the `TotalEnergy`.
    The inheritance from `PropertyContribution` allows to link this contribution to a
    specific component (of class `BaseModelMethod`) of the over `ModelMethod` using the
    `model_method_ref` quantity.

    For example, for a force field calculation, the `model_method_ref` may point to a
    particular potential type (e.g., a Lennard-Jones potential between atom types X and Y),
    while for a DFT calculation, it may point to a particular electronic interaction term
    (e.g., 'XC' for the exchange-correlation term, or 'Hartree' for the Hartree term).
    Then, the contribution will be named according to this model component and the `value`
    quantity will contain the energy contribution from this component evaluated over all
    relevant atoms or electrons or as a function of them.
    """

    # TODO address the dual parent normalization explicity
    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


####################################
# List of specific energy properties
####################################


class FermiLevel(BaseEnergy):
    """
    Energy required to add or extract a charge from a material at zero temperature. It can be also defined as the chemical potential at zero temperature.
    """

    # ! implement `iri` and `rank` as part of `m_def = Section()`

    iri = 'http://fairmat-nfdi.eu/taxonomy/FermiLevel'

    def __init__(
        self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.rank = []
        self.name = self.m_def.name

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


#! The only issue with this structure is that total energy will never be a sum of its contributions,
#! since kinetic energy lives separately, but I think maybe this is ok?
class TotalEnergy(BaseEnergy):
    """
    The total energy of a system. `contributions` specify individual energetic
    contributions to the `TotalEnergy`.
    """

    # ? add a generic contributions quantity to PhysicalProperty
    contributions = SubSection(sub_section=EnergyContribution.m_def, repeats=True)

    def __init__(
        self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.name = self.m_def.name

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


# ? Separate quantities for nuclear and electronic KEs?
class KineticEnergy(BaseEnergy):
    """
    Physical property section describing the kinetic energy of a (sub)system.
    """

    def __init__(
        self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.name = self.m_def.name

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class PotentialEnergy(BaseEnergy):
    """
    Physical property section describing the potential energy of a (sub)system.
    """

    def __init__(
        self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.name = self.m_def.name

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)
