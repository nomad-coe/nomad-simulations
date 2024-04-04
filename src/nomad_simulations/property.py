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

import re
import numpy as np

from nomad.units import ureg
from nomad.datamodel.data import ArchiveSection
from nomad.metainfo import Quantity, SubSection, Reference
from nomad.metainfo.metainfo import DirectQuantity, Dimension
from nomad.datamodel.metainfo.annotations import ELNAnnotation

from .model_system import ModelSystem
from .outputs import Outputs

class Errors(ArchiveSection):
    """
    A base section used to define errors.
    """

    type = Quantity(
        type=str,
        description="""
        Type of error.
        """,
        a_eln=ELNAnnotation(component='StringEditQuantity'),
    )

    value = Quantity(
        type=np.float64,
        shape=['*'],
        description="""
        Value/s of the error.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class Smoothing(ArchiveSection):
    """
    A base section used to define data smoothing procedures.
    """

    type = Quantity(
        type=str,
        description="""
        Type of smoothing, e.g., "running_average".
        """,
        a_eln=ELNAnnotation(component='StringEditQuantity'),
    )

    parameters = SubSection(sub_section=ParameterEntry.m_def, repeats=True)

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class PhysicalPropery(ArchiveSection):
    """
    A base section used to define properties.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    name = Quantity(
        type=str,
        description="""
        Name of the Property section. This will be set by the individual subclasses.
        """,
        a_eln=ELNAnnotation(component='StringEditQuantity'),
    )

    type = Quantity(
        type=str,
        description="""
        Categorization of the property.
        """,
        a_eln=ELNAnnotation(component='StringEditQuantity'),
    )

    label = Quantity(
        type=str,
        shape=[],
        description="""
        Additional descriptive label for the property.
        """,
    )

    description = Quantity(
        type=str,
        shape=[],
        description="""
        Detailed description of the property.
        """,
    )

    # ? Just copied this from Quantity, not sure what I am doing
    # shape/fullshape = DirectQuantity(type=Dimension, shape=['0..*'], name='shape', default=[])

    variables = Quantity(
        type=str,
        shape=['*'],
        description="""
        Name/description of the variables along which the property is defined.
        """,
    )

    bins = Quantity(
        type=np.float64,
        shape=['*'],
        description="""
        Values of the variable along which the property is stored.
        """,
    )

    # TODO Add a check in normalization to ensure that value and bins have the same first dimension
    value = Quantity(
        type=np.float64,
        shape=['*'],
        description="""
        Values of the property with units defined within the individual subclass.
        """,
    )

    # TODO Add value_per_particle?

    # ! Probably not needed, already in Quantity
    is_scalar = Quantity(
        type=bool,
        default=False,
        description="""
        Flag indicating whether the output property is a scalar. If yes, variable and bin quantities
        are ensured to not be populated.
        """,
    )

    errors = SubSection(sub_section=Errors.m_def, repeats=True)
    smoothing = SubSection(sub_section=Smoothing.m_def, repeats=True)

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        # ! Use is scalar under value instead?
        if self.is_scalar:
            if self.variables is not None:
                self.variables = None
                # TODO throw warning/error
            if self.bins is not None:
                self.bins = None
                # TODO throw warning/error
                # TODO ...

