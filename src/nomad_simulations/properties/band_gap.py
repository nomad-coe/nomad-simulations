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
import pint
from typing import Optional

from nomad.units import ureg
from nomad.metainfo import Quantity, MEnum, Section, Context

from ..physical_property import PhysicalProperty


class ElectronicBandGap(PhysicalProperty):
    """
    Energy difference between the highest occupied electronic state and the lowest unoccupied electronic state.
    """

    iri = 'https://fairmat-nfdi.github.io/fairmat-taxonomy/#/ElectronicBandGap'

    type = Quantity(
        type=MEnum('direct', 'indirect'),
        description="""
        Type categorization of the `ElectronicBandGap`. This quantity is directly related with `momentum_transfer` as by
        definition, the electronic band gap is `'direct'` for zero momentum transfer (or if `momentum_transfer` is `None`) and `'indirect'`
        for finite momentum transfer.
        """,
    )

    momentum_transfer = Quantity(
        type=np.float64,
        shape=[2, 3],
        description="""
        If the `ElectronicBandGap` is `'indirect'`, the reciprocal momentum transfer for which the band gap is defined
        in units of the `reciprocal_lattice_vectors`. The initial and final momentum 3D vectors are given in the first
        and second element. Example, the momentum transfer in bulk Si2 happens between the Γ and the (approximately)
        X points in the Brillouin zone; thus:
            `momentum_transfer = [[0, 0, 0], [0.5, 0.5, 0]]`.
        """,
    )

    spin_channel = Quantity(
        type=np.int32,
        description="""
        Spin channel of the corresponding `ElectronicBandGap`. It can take values of 0 or 1.
        """,
    )

    value = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        The value of the `ElectronicBandGap`.
        """,
    )

    def __init__(self, m_def: Section = None, m_context: Context = None, **kwargs):
        super().__init__(m_def, m_context, **kwargs)
        self.name = self.m_def.name
        self.rank = []  # ? Is this here or in the attrs instantiation better?

    def check_negative_values(self, logger: BoundLogger) -> pint.Quantity:
        """
        Checks if the electronic band gap is negative and sets it to 0 if it is.

        Args:
            logger (BoundLogger): The logger to log messages.
        """
        if self.value < 0.0:
            logger.error(
                'The electronic band gap cannot be defined negative. We set it up to 0.'
            )
            return 0.0 * ureg.eV
        return self.value

    def resolve_type(self, logger: BoundLogger) -> Optional[str]:
        """
        Resolves the `type` of the electronic band gap based on the stored `momentum_transfer` values.

        Args:
            logger (BoundLogger): The logger to log messages.

        Returns:
            (Optional[str]): The resolved `type` of the electronic band gap.
        """
        if self.momentum_transfer is None and self.type == 'indirect':
            logger.warning(
                "The `momentum_transfer` is not defined for an `type='indirect'` electronic band gap."
            )
            return None
        if self.momentum_transfer is not None:
            momentum_difference = np.diff(self.momentum_transfer, axis=0)
            if (momentum_difference == np.zeros(3)).all():
                return 'direct'
            else:
                return 'indirect'
        return self.type

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        # Checks if the `value` is negative and sets it to 0 if it is.
        self.value = self.check_negative_values(logger)

        # Resolve the `type` of the electronic band gap.
        self.type = self.resolve_type(logger)
