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
#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD.
# See https://nomad-lab.eu for further info.
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

from nomad.metainfo import Quantity, Section, Context, SubSection, MEnum
from nomad.datamodel.data import ArchiveSection

from nomad_simulations.physical_property import PhysicalProperty

####################################################
# Abstract force classes
####################################################

class Force(PhysicalProperty):
    """
    Abstract physical property section describing some energy of a (sub)system.
    """

    type = Quantity(
        type=MEnum('classical', 'quantum'),
        description="""
        """,
    )

    value = Quantity(
        type=np.dtype(np.float64),
        unit='newton',
        description="""
        The value of the force.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

######################################################
# List of general force properties/contributions that
# can have both classical and quantum interpretations
######################################################

class TotalForce(Force):
    """
    Section containing the total force of a (sub)system.

    Contains the value and information regarding the total forces on the atoms
    calculated as minus gradient of energy_total.
    """
    # ! We need to avoid giving the precise method of calculation without also providing context, this is not necessarily true in general!

    contributions = SubSection(
        sub_section=Force.m_def, repeats=True
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


################################
# List of Forces Contributions #
################################


class FreeForce(Force):
    """
    Physical property section describing...

    Contains the value and information regarding the forces on the atoms
    corresponding to the minus gradient of energy_free. The (electronic) energy_free
    contains the information on the change in (fractional) occupation of the
    electronic eigenstates, which are accounted for in the derivatives, yielding a
    truly energy-conserved quantity.
    """

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class ZeroTemperatureForce(Force):
    """
    Physical property section describing...

    Contains the value and information regarding the forces on the atoms
    corresponding to the minus gradient of energy_T0.
    """

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class RawForce(Force):
    """
    Physical property section describing...

    Value of the forces acting on the atoms **not including** such as fixed atoms,
    distances, angles, dihedrals, etc.
    """
    # ? This is VERY imprecise, is this used regularly?

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


# ? Do we want to support custom contributions?
# contributions = SubSection(
#     sub_section=ForcesEntry.m_def,
#     description="""
#     Contains other forces contributions to the total atomic forces not already
#     defined.
#     """,
#     repeats=True,
# )

# types = SubSection(
#     sub_section=ForcesEntry.m_def,
#     description="""
#     Contains other types of forces not already defined.
#     """,
#     repeats=True,
# )


# Old version of the Forces description
# Value of the forces acting on the atoms. This is calculated as minus gradient of
# the corresponding energy type or contribution **including** constraints, if
# present. The derivatives with respect to displacements of nuclei are evaluated in
# Cartesian coordinates.  In addition, these are obtained by filtering out the
# unitary transformations (center-of-mass translations and rigid rotations for
# non-periodic systems, see value_raw for the unfiltered counterpart).

