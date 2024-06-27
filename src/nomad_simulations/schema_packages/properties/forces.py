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

from nomad.metainfo import Quantity, Section, Context, SubSection
from nomad_simulations.schema_packages.physical_property import (
    PhysicalProperty,
    PropertyContribution,
)

####################################################
# Abstract force classes
####################################################

##################
# Abstract classes
##################


class BaseForce(PhysicalProperty):
    """
    Abstract physical property section describing some force of a (sub)system.
    """

    value = Quantity(
        type=np.dtype(np.float64),
        unit='newton',
        description="""
        The value of the force.
        """,
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class ForceContribution(BaseForce, PropertyContribution):
    """
    Abstract physical property section linking a property contribution to a contribution
    from some method.
    """

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


###################################
# List of specific force properties
###################################


class TotalForce(BaseForce):
    """
    Physical property section describing the total force of a (sub)system.
    """

    contributions = SubSection(sub_section=ForceContribution.m_def, repeats=True)

    def __init__(
        self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.name = self.m_def.name

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


# ? See questions about corresponding energies
# class FreeForce(Force):
#     """
#     Physical property section describing...

#     Contains the value and information regarding the forces on the atoms
#     corresponding to the minus gradient of energy_free. The (electronic) energy_free
#     contains the information on the change in (fractional) occupation of the
#     electronic eigenstates, which are accounted for in the derivatives, yielding a
#     truly energy-conserved quantity.
#     """

#     def normalize(self, archive, logger) -> None:
#         super().normalize(archive, logger)


# class ZeroTemperatureForce(Force):
#     """
#     Physical property section describing...

#     Contains the value and information regarding the forces on the atoms
#     corresponding to the minus gradient of energy_T0.
#     """

#     def normalize(self, archive, logger) -> None:
#         super().normalize(archive, logger)
