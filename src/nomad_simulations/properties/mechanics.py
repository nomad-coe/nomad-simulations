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
from nomad.metainfo import Quantity, SubSection, SectionProxy, MEnum
from nomad.metainfo.metainfo import DirectQuantity, Dimension
from nomad.datamodel.metainfo.annotations import ELNAnnotation

from .outputs import Outputs, SCFOutputs, WorkflowOutputs, TrajectoryOutputs
from .atoms_state import AtomsState
from .physical_property import PhysicalProperty


from nomad.datamodel.metainfo.common import (
    FastAccess,
    PropertySection,
    ProvenanceTracker,
)
from nomad.metainfo import (
    Category,
    HDF5Reference,
    MCategory,
    MEnum,
    MSection,
    Package,
    Quantity,
    Reference,
    Section,
    SectionProxy,
    SubSection,
)

from .model_system import ModelSystem


class StressEntry(Atomic):
    """
    Section describing a contribution to or a type of stress.
    """

    m_def = Section(validate=False)

    value = Quantity(
        type=np.dtype(np.float64),
        shape=[3, 3],
        unit='joule/meter**3',
        description="""
        Value of the stress on the simulation cell. It is given as the functional
        derivative of the corresponding energy with respect to the deformation tensor.
        """,
    )

    values_per_atom = Quantity(
        type=np.dtype(np.float64),
        shape=['number_of_atoms', 3, 3],
        unit='joule/meter**3',
        description="""
        Value of the atom-resolved stresses.
        """,
    )


class Stress(MSection):
    """
    Section containing all stress types and contributions.
    """

    m_def = Section(validate=False)

    total = SubSection(
        sub_section=StressEntry.m_def,
        description="""
        Contains the value and information regarding the stress on the simulation cell
        and the atomic stresses corresponding to energy_total.
        """,
    )

    contributions = SubSection(
        sub_section=StressEntry.m_def,
        description="""
        Contains contributions for the total stress.
        """,
        repeats=True,
    )

    types = SubSection(
        sub_section=StressEntry.m_def,
        description="""
        Contains other types of stress.
        """,
        repeats=True,
    )