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

from nomad.metainfo import Quantity, Section, Context, SubSection

if TYPE_CHECKING:
    from nomad.metainfo import Section, Context
    from nomad.datamodel.datamodel import EntryArchive
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
    Abstract physical property section describing some energy of a (sub)system.
    """

    value = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        The value of the energy.
        """,
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class EnergyContribution(BaseEnergy, PropertyContribution):
    """
    Abstract physical property section linking a property contribution to a contribution
    from some method.
    """

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


####################################
# List of specific energy properties
####################################


class FermiLevel(BaseEnergy):
    """
    Physical property section describing the Fermi level, i.e., the energy required to add or extract a charge from a material at zero temperature.
    It can be also defined as the chemical potential at zero temperature.
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
    Physical property section describing the total energy of a (sub)system.
    """

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

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class PotentialEnergy(BaseEnergy):
    """
    Physical property section describing the potential energy of a (sub)system.
    """

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


#! I removed all previous contributions associated in some way with terms in the Hamiltonian.
# ? Should the remaining contributions below be incorporated into some sort of workflow results if still relevant?


# class ZeroTemperatureEnergy(QuantumEnergy):
#     """
#     Physical property section describing the total energy of a (sub)system extrapolated to $T=0$, based on a free-electron gas argument.
#     """

#     def normalize(self, archive, logger) -> None:
#         super().normalize(archive, logger)


# class ZeroPointEnergy(QuantumEnergy):
#     """
#     Physical property section describing the zero-point vibrational energy of a (sub)system,
#     calculated using the method described in zero_point_method.
#     """

#     def normalize(self, archive, logger) -> None:
#         super().normalize(archive, logger)


# madelung = SubSection(
#     sub_section=EnergyEntry.m_def,
#     description="""
#     Contains the value and information regarding the Madelung energy.
#     """,
# )

# free = SubSection(
#     sub_section=EnergyEntry.m_def,
#     description="""
#     Contains the value and information regarding the free energy (nuclei + electrons)
#     (whose minimum gives the smeared occupation density calculated with
#     smearing_kind).
#     """,
# )

# sum_eigenvalues = SubSection(
#     sub_section=EnergyEntry.m_def,
#     description="""
#     Contains the value and information regarding the sum of the eigenvalues of the
#     Hamiltonian matrix.
#     """,
# )
