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

from nomad_simulations.numerical_settings import (
    KMesh as KMeshSettings,
    KLinePath as KLinePathSettings,
)


class Variables(ArchiveSection):
    """
    Variables over which the physical property varies, and they are defined as grid points, i.e., discretized
    values by `n_points` and `points`. These are used to calculate the `shape` of the physical property.
    """

    name = Quantity(
        type=str,
        default='Custom',
        description="""
        Name of the variable.
        """,
    )

    n_points = Quantity(
        type=int,
        description="""
        Number of points in which the variable is discretized.
        """,
    )

    points = Quantity(
        type=np.float64,
        shape=['n_points'],
        description="""
        Points in which the variable is discretized. It might be overwritten with specific units.
        """,
    )

    # points_error = Quantity()

    def get_n_points(self, logger: BoundLogger) -> Optional[int]:
        """
        Get the number of grid points from the `points` list. If `n_points` is previously defined
        and does not coincide with the length of `points`, a warning is issued and this function re-assigns `n_points`
        as the length of `points`.

        Args:
            logger (BoundLogger): The logger to log messages.

        Returns:
            (Optional[int]): The number of points.
        """
        if self.points is not None and len(self.points) > 0:
            if self.n_points != len(self.points) and self.n_points is not None:
                logger.warning(
                    f'The stored `n_points`, {self.n_points}, does not coincide with the length of `points`, '
                    f'{len(self.points)}. We will re-assign `n_points` as the length of `points`.'
                )
            return len(self.points)
        return self.n_points

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        # Setting `n_points` if these are not defined
        self.n_points = self.get_n_points(logger)


class Temperature(Variables):
    """ """

    points = Quantity(
        type=np.float64,
        unit='kelvin',
        shape=['n_points'],
        description="""
        Points in which the temperature is discretized.
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

    points = Quantity(
        type=np.float64,
        unit='joule',
        shape=['n_points'],
        description="""
        Points in which the energy is discretized.
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

    points = Quantity(
        type=np.float64,
        shape=['n_points', 3],
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


class KMesh(Variables):
    """
    K-point mesh over which the physical property is calculated. This is used to define `ElectronicEigenvalues(PhysicalProperty)` and
    other k-space properties. The `points` are obtained from a refernece to the `NumericalSettings` section, `KMesh(NumericalSettings)`.
    """

    k_mesh_settings_ref = Quantity(
        type=KMeshSettings,
        description="""
        Reference to the `KMesh(NumericalSettings)` section in the `ModelMethod` section. This reference is useful
        to extract `points` and, then, obtain the shape of `value` of the `PhysicalProperty`.
        """,
    )

    points = Quantity(
        type=np.float64,
        shape=['n_points', 'dimensionality'],
        description="""
        K-point mesh over which the physical property is calculated. These are 3D arrays stored in fractional coordinates.
        """,
    )

    def __init__(
        self, m_def: Section = None, m_context: Context = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.name = self.m_def.name

    def extract_points(self, logger: BoundLogger) -> Optional[list]:
        """
        Extract the `points` list from the `k_mesh_settings_ref` pointing to the `KMesh` section.
        Args:
            logger (BoundLogger): The logger to log messages.
        Returns:
            (Optional[list]): The `points` list.
        """
        if self.k_mesh_settings_ref is not None:
            if self.k_mesh_settings_ref.points is not None:
                return self.k_mesh_settings_ref.points
            points, _ = self.k_mesh_settings_ref.resolve_points_and_offset(logger)
            return points
        logger.error('`k_mesh_settings_ref` is not defined.')
        return None

    def normalize(self, archive, logger) -> None:
        # Extracting `points` from the `k_mesh_settings_ref` BEFORE doing `super().normalize()`
        self.points = self.extract_points(logger)

        super().normalize(archive, logger)


class KLinePath(Variables):
    """ """

    k_line_path_settings_ref = Quantity(
        type=KLinePathSettings,
        description="""
        Reference to the `KLinePath(NumericalSettings)` section in the `ModelMethod.KMesh` section. This reference is useful
        to extract `points` and, then, obtain the shape of `value` of the `PhysicalProperty`.
        """,
    )

    points = Quantity(
        type=np.float64,
        shape=['n_points', 3],
        description="""
        Points along the k-line path in which the physical property is calculated. These are 3D arrays stored in fractional coordinates.
        """,
    )

    def __init__(
        self, m_def: Section = None, m_context: Context = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.name = self.m_def.name

    def extract_points(self, logger: BoundLogger) -> Optional[list]:
        """
        Extract the `points` list from the `k_line_path_settings_ref` pointing to the `KLinePath` section.
        Args:
            logger (BoundLogger): The logger to log messages.
        Returns:
            (Optional[list]): The `points` list.
        """
        if self.k_line_path_settings_ref is not None:
            return self.k_line_path_settings_ref.points
        logger.error('`k_line_path_settings_ref` is not defined.')
        return None

    def normalize(self, archive, logger) -> None:
        # Extracting `points` from the `k_line_path_settings_ref` BEFORE doing `super().normalize()`
        self.points = self.extract_points(logger)

        super().normalize(archive, logger)
