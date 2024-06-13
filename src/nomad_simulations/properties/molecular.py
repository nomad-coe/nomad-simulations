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


class RadiusOfGyration(PhysicalProperty):

    value = Quantity(
        type=np.float64,
        unit='m',
    )

class Hessian(TrajectoryOutputs):
    """
    Section containing the Hessian matrix, i.e., 2nd derivatives with respect to atomic displacements,
    of the potential energy of a (sub)system.
    """

    value = Quantity(
        type=np.dtype(np.float64),
        shape=['n_atoms', 'n_atoms', 3, 3],
        unit='joule / m ** 2',
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)
        self.value_unit = 'joule / m ** 2'
        self.name = 'hessian'
        self.variables = [['atom_index'], ['atom_index'], ['x', 'y', 'z'], ['x', 'y', 'z']]
        self.bins = [np.range(self.n_atoms), np.range(self.n_atoms)]


class RadialDistributionFunction(TrajectoryOutputs):
    """
    Section containing information about the calculation of
    radial distribution functions (rdfs).
    """

    value = Quantity(
        type=np.dtype(np.float64),
        shape=['*'],
    )

    variables = Quantity(
        type=str,
        shape=[1],
        description="""
        Name/description of the variables along which the property is defined.
        """,
    )

    bins = Quantity(
        type=np.float64,
        shape=['*'],
        unit='m',
        description="""
        Distances along which the rdf was calculated.
        """,
    )

    def normalize(self, archive, logger):
        super().normalize(archive, logger)
        self.variables = ['distance']
        assert len(self.bins) == len(self.value)
