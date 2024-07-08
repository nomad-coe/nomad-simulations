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

from typing import TYPE_CHECKING, Optional

import numpy as np
import pint
from nomad.metainfo import MEnum, Quantity

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from nomad.metainfo import Context, Section
    from structlog.stdlib import BoundLogger

from nomad_simulations.schema_packages.physical_property import PhysicalProperty


class ElectronicBandGap(PhysicalProperty):
    """
    Energy difference between the highest occupied electronic state and the lowest unoccupied electronic state.
    """

    # ! implement `iri` and `rank` as part of `m_def = Section()`

    iri = 'http://fairmat-nfdi.eu/taxonomy/ElectronicBandGap'

    type = Quantity(
        type=MEnum('direct', 'indirect'),
        description="""
        Type categorization of the electronic band gap. This quantity is directly related with `momentum_transfer` as by
        definition, the electronic band gap is `'direct'` for zero momentum transfer (or if `momentum_transfer` is `None`) and `'indirect'`
        for finite momentum transfer.

        Note: in the case of finite `variables`, this quantity refers to all of the `value` in the array.
        """,
    )

    momentum_transfer = Quantity(
        type=np.float64,
        shape=[2, 3],
        description="""
        If the electronic band gap is `'indirect'`, the reciprocal momentum transfer for which the band gap is defined
        in units of the `reciprocal_lattice_vectors`. The initial and final momentum 3D vectors are given in the first
        and second element. Example, the momentum transfer in bulk Si2 happens between the Γ and the (approximately)
        X points in the Brillouin zone; thus:
            `momentum_transfer = [[0, 0, 0], [0.5, 0.5, 0]]`.

        Note: this quantity only refers to scalar `value`, not to arrays of `value`.
        """,
    )

    spin_channel = Quantity(
        type=np.int32,
        description="""
        Spin channel of the corresponding electronic band gap. It can take values of 0 or 1.
        """,
    )

    value = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        The value of the electronic band gap. This value has to be positive, otherwise it will
        prop an error and be set to None by the `normalize()` function.
        """,
    )

    def __init__(
        self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.name = self.m_def.name
        self.rank = []

    def validate_values(self, logger: 'BoundLogger') -> Optional[pint.Quantity]:
        """
        Validate the electronic band gap `value` by checking if they are negative and sets them to None if they are.

        Args:
            logger (BoundLogger): The logger to log messages.
        """
        value = self.value.magnitude
        if not isinstance(self.value.magnitude, np.ndarray):  # for scalars
            value = np.array(
                [value]
            )  # ! check this when talking with Lauri and Theodore

        # Set the value to 0 when it is negative
        if (value < 0).any():
            logger.error('The electronic band gap cannot be defined negative.')
            return None

        if not isinstance(self.value.magnitude, np.ndarray):  # for scalars
            value = value[0]
        return value * self.value.u

    def resolve_type(self, logger: 'BoundLogger') -> Optional[str]:
        """
        Resolves the `type` of the electronic band gap based on the stored `momentum_transfer` values.

        Args:
            logger (BoundLogger): The logger to log messages.

        Returns:
            (Optional[str]): The resolved `type` of the electronic band gap.
        """
        mtr = self.momentum_transfer if self.momentum_transfer is not None else []

        # Check if the `momentum_transfer` is [], and return the type and a warning in the log for `indirect` band gaps
        if len(mtr) == 0:
            if self.type == 'indirect':
                logger.warning(
                    'The `momentum_transfer` is not stored for an `indirect` band gap.'
                )
            return self.type

        # Check if the `momentum_transfer` has at least two elements, and return None if it does not
        if len(mtr) == 1:
            logger.warning(
                'The `momentum_transfer` should have at least two elements so that the difference can be calculated and the type of electronic band gap can be resolved.'
            )
            return None

        # Resolve `type` from the difference between the initial and final momentum transfer
        momentum_difference = np.diff(mtr, axis=0)
        if (np.isclose(momentum_difference, np.zeros(3))).all():
            return 'direct'
        else:
            return 'indirect'

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

        # Checks if the `value` is negative and sets it to None if it is.
        self.value = self.validate_values(logger)
        if self.value is None:
            # ? What about deleting the class if `value` is None?
            logger.error('The `value` of the electronic band gap is not stored.')
            return

        # Resolve the `type` of the electronic band gap from `momentum_transfer`, ONLY for scalar `value`
        if isinstance(self.value.magnitude, np.ndarray):
            logger.info(
                'We do not support `type` which describe individual elements in an array `value`.'
            )
        else:
            self.type = self.resolve_type(logger)
