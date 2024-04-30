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
from typing import Optional
from structlog.stdlib import BoundLogger

from nomad.datamodel.data import ArchiveSection
from nomad.metainfo import Quantity, Section, Context


class Variables(ArchiveSection):
    """
    Variables over which the physical property varies. These are defined as binned, i.e., discretized
    values by `n_bins` and `bins`. These are used to calculate the `shape` of the physical property.
    """

    name = Quantity(
        type=str,
        default='Custom',
        description="""
        Name of the variable.
        """,
    )

    n_grid_points = Quantity(
        type=int,
        description="""
        Number of grid points in which the variable is discretized.
        """,
    )

    grid_points = Quantity(
        type=np.float64,
        shape=['n_grid_points'],
        description="""
        Grid points in which the variable is discretized. It might be overwritten with specific units.
        """,
    )

    # grid_points_error = Quantity()

    def get_n_grid_points(self, logger: BoundLogger) -> Optional[int]:
        """
        Get the number of grid points from the `grid_points` list. If `n_grid_points` is previously defined
        and does not coincide with the length of `grid_points`, a warning is issued and this function re-assigns `n_grid_points`
        as the length of `grid_points`.

        Args:
            logger (BoundLogger): The logger to log messages.

        Returns:
            (Optional[int]): The number of grid points.
        """
        if self.grid_points is not None and len(self.grid_points) > 0:
            if (
                self.n_grid_points != len(self.grid_points)
                and self.n_grid_points is not None
            ):
                logger.warning(
                    f'The stored `n_grid_points`, {self.n_grid_points}, does not coincide with the length of `grid_points`, '
                    f'{len(self.grid_points)}. We will re-assign `n_grid_points` as the length of `grid_points`.'
                )
            return len(self.grid_points)
        return self.n_grid_points

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        # Setting `n_grid_points` if these are not defined
        self.n_grid_points = self.get_n_grid_points(logger)


class Temperature(Variables):
    """ """

    grid_points = Quantity(
        type=np.float64,
        unit='kelvin',
        shape=['n_grid_points'],
        description="""
        Grid points in which the temperature is discretized.
        """,
    )

    def __init__(
        self, m_def: Section = None, m_context: Context = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.name = self.m_def.name

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


# ! This needs to be fixed as it gives errors when running normalizers with conflicting names (ask Area D)
class Energy2(Variables):
    """ """

    grid_points = Quantity(
        type=np.float64,
        unit='joule',
        shape=['n_grid_points'],
        description="""
        Grid points in which the energy is discretized.
        """,
    )

    def __init__(
        self, m_def: Section = None, m_context: Context = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.name = self.m_def.name

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class WignerSeitz(Variables):
    """
    Wigner-Seitz points in which the real space is discretized. This variable is used to define `HoppingMatrix(PhysicalProperty)` and
    other inter-cell properties. See, e.g., https://en.wikipedia.org/wiki/Wigner–Seitz_cell.
    """

    grid_points = Quantity(
        type=np.float64,
        shape=['n_grid_points', 3],
        description="""
        Wigner-Seitz points with respect to the origin cell, (0, 0, 0). These are 3D arrays stored in fractional coordinates.
        """,
    )

    def __init__(
        self, m_def: Section = None, m_context: Context = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.name = self.m_def.name

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)
