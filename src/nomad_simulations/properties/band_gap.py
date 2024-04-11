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

from nomad.metainfo import Quantity, MEnum, Section, Context

from ..physical_property import PhysicalProperty


class ElectronicBandGap(PhysicalProperty):
    """ """

    type = Quantity(
        type=MEnum('direct', 'indirect'),
        description="""
        Type categorization of the electronic band gap. The electronic band gap can be `'direct'` or `'indirect'`.
        """,
    )

    value = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        The value of the electronic band gap.
        """,
    )

    # TODO add more functionalities here

    def __init__(self, m_def: Section = None, m_context: Context = None, **kwargs):
        super().__init__(m_def, m_context, **kwargs)
        self.name = self.m_def.name
        self.rank = []

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)
