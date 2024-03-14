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
import pint
from structlog.stdlib import BoundLogger
from typing import Optional, List, Tuple
from ase.dft.kpoints import monkhorst_pack, get_monkhorst_pack_size_and_offset

from nomad.units import ureg
from nomad.datamodel.data import ArchiveSection
from nomad.metainfo import (
    Quantity,
    SubSection,
    MEnum,
    Section,
    Context,
    JSON,
)

from .model_system import ModelSystem
from .utils import is_not_representative


class NumericalSettings(ArchiveSection):
    """
    A base section used to define the numerical settings used in a simulation. These are meshes,
    self-consistency parameters, and basis sets.
    """

    name = Quantity(
        type=str,
        description="""
        Name of the numerical settings section. This is typically used to easy identification of the
        `NumericalSettings` section. Possible values: "KMesh", "FrequencyMesh", "TimeMesh",
        "SelfConsistency", "BasisSet".
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class Mesh(NumericalSettings):
    """
    A base section used to specify the settings of a sampling mesh. It supports uniformly-spaced
    meshes and symmetry-reduced representations.
    """

    spacing = Quantity(
        type=MEnum('Equidistant', 'Logarithmic', 'Tan'),
        default='Equidistant',
        description="""
        Identifier for the spacing of the Mesh. Defaults to 'Equidistant' if not defined. It can take the values:

        | Name      | Description                      |
        | --------- | -------------------------------- |
        | `'Equidistant'`  | Equidistant grid (also known as 'Newton-Cotes') |
        | `'Logarithmic'`  | log distance grid |
        | `'Tan'`  | Non-uniform tan mesh for grids. More dense at low abs values of the points, while less dense for higher values |
        """,
    )

    center = Quantity(
        type=MEnum('Gamma-centered', 'Monkhorst-Pack', 'Gamma-offcenter'),
        description="""
        Identifier for the center of the Mesh:

        | Name      | Description                      |
        | --------- | -------------------------------- |
        | `'Gamma-centered'` | Regular mesh is centered around Gamma. No offset. |
        | `'Monkhorst-Pack'` | Regular mesh with an offset of half the reciprocal lattice vector. |
        | `'Gamma-offcenter'` | Regular mesh with an offset that is neither `'Gamma-centered'`, nor `'Monkhorst-Pack'`. |
        """,
    )

    quadrature = Quantity(
        type=MEnum(
            'Gauss-Legendre',
            'Gauss-Laguerre',
            'Clenshaw-Curtis',
            'Newton-Cotes',
            'Gauss-Hermite',
        ),
        description="""
        Quadrature rule used for integration of the Mesh. This quantity is relevant for 1D meshes:

        | Name      | Description                      |
        | --------- | -------------------------------- |
        | `'Gauss-Legendre'` | Quadrature rule for integration using Legendre polynomials |
        | `'Gauss-Laguerre'` | Quadrature rule for integration using Laguerre polynomials |
        | `'Clenshaw-Curtis'`  | Quadrature rule for integration using Chebyshev polynomials using discrete cosine transformations |
        | `'Gauss-Hermite'`  | Quadrature rule for integration using Hermite polynomials |
        """,
    )

    n_points = Quantity(
        type=np.int32,
        description="""
        Number of points in the mesh.
        """,
    )

    dimensionality = Quantity(
        type=np.int32,
        default=3,
        description="""
        Dimensionality of the mesh: 1, 2, or 3. If not defined, it is assumed to be 3.
        """,
    )

    grid = Quantity(
        type=np.int32,
        shape=['dimensionality'],
        description="""
        Amount of mesh point sampling along each axis, i.e. [nx, ny, nz].
        """,
    )

    points = Quantity(
        type=np.complex128,
        shape=['n_points', 'dimensionality'],
        description="""
        List of all the points in the mesh.
        """,
    )

    multiplicities = Quantity(
        type=np.float64,
        shape=['n_points'],
        description="""
        The amount of times the same point reappears. A value larger than 1, typically indicates
        a symmtery operation that was applied to the `Mesh`. This quantity is equivalent to `weights`:

            multiplicities = 1 / weights
        """,
    )

    weights = Quantity(
        type=np.float64,
        shape=['n_points'],
        description="""
        Weight of each point. A value smaller than 1, typically indicates a symmtery operation that was
        applied to the mesh. This quantity is equivalent to `multiplicities`:

            weights = 1 / multiplicities
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class LinePathSegment(ArchiveSection):
    """
    A base section used to define the settings of a single line path segment within a multidimensional mesh.
    """

    high_symmetry_path = Quantity(
        type=str,
        shape=[2],
        description="""
        List of the two high-symmetry points followed in the line path segment, e.g., ['Gamma', 'X']. The
        point's coordinates can be extracted from the values in the `self.m_parent.high_symmetry_points` JSON quantity.
        """,
    )

    n_line_points = Quantity(
        type=np.int32,
        description="""
        Number of points in the line path segment.
        """,
    )

    points = Quantity(
        type=np.float64,
        shape=['n_line_points', 3],
        description="""
        List of all the points in the line path segment in units of the `reciprocal_lattice_vectors`.
        """,
    )

    def resolve_points(
        self,
        high_symmetry_path: List[str],
        n_line_points: int,
        logger: BoundLogger,
    ) -> Optional[np.ndarray]:
        """
        Resolves the `points` of the `LinePathSegment` from the `high_symmetry_path` and the `n_line_points`.

        Args:
            high_symmetry_path (List[str]): The high-symmetry path of the `LinePathSegment`.
            n_line_points (int): The number of points in the line path segment.
            logger (BoundLogger): The logger to log messages.

        Returns:
            (Optional[List[np.ndarray]]): The resolved `points` of the `LinePathSegment`.
        """
        if high_symmetry_path is None or n_line_points is None:
            logger.warning(
                'Could not resolve `LinePathSegment.points` from `LinePathSegment.high_symmetry_path` and `LinePathSegment.n_line_points`.'
            )
            return None
        if self.m_parent.high_symmetry_points is None:
            logger.warning(
                'Could not resolve the parent of `LinePathSegment` to extract `LinePathSegment.m_parent.high_symmetry_points`.'
            )
            return None
        start_point = self.m_parent.high_symmetry_points.get(self.high_symmetry_path[0])
        end_point = self.m_parent.high_symmetry_points.get(self.high_symmetry_path[1])
        return np.linspace(start_point, end_point, self.n_line_points)

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        if self.points is None:
            self.points = self.resolve_points(
                self.high_symmetry_path, self.n_line_points, logger
            )


class KMesh(Mesh):
    """
    A base section used to specify the settings of a sampling mesh in reciprocal space.
    """

    reciprocal_lattice_vectors = Quantity(
        type=np.float64,
        shape=[3, 3],
        unit='1/meter',
        description="""
        Reciprocal lattice vectors of the simulated cell, in Cartesian coordinates and
        including the $2 pi$ pre-factor. The first index runs over each lattice vector. The
        second index runs over the $x, y, z$ Cartesian coordinates.
        """,
    )

    offset = Quantity(
        type=np.float64,
        shape=[3],
        description="""
        Offset vector shifting the mesh with respect to a Gamma-centered case (where it is defined as [0, 0, 0]).
        """,
    )

    all_points = Quantity(
        type=np.float64,
        shape=['*', 3],
        description="""
        Full list of the mesh points without any symmetry operations. In the presence of symmetry
        operations, this quantity is a larger list than `points` (as it will contain all the points in the Brillouin zone).
        """,
    )

    high_symmetry_points = Quantity(
        type=JSON,
        description="""
        Dictionary containing the high-symmetry points and their points in terms of `reciprocal_lattice_vectors`.
        E.g., in a cubic lattice:

            high_symmetry_points = {
                'Gamma': [0, 0, 0],
                'X': [0.5, 0, 0],
            }
        """,
    )

    k_line_density = Quantity(
        type=np.float64,
        unit='m',
        description="""
        Amount of sampled k-points per unit reciprocal length along each axis.
        Contains the least precise density out of all axes.
        Should only be compared between calulations of similar dimensionality.
        """,
    )

    line_path_segments = SubSection(sub_section=LinePathSegment.m_def, repeats=True)

    # TODO add extraction of `high_symmetry_points` using BandStructureNormalizer idea (left for later when defining outputs.py)

    def __init__(self, m_def: Section = None, m_context: Context = None, **kwargs):
        super().__init__(m_def, m_context, **kwargs)
        # Set the name of the section
        self.name = self.m_def.name

    def resolve_points_and_offset(
        self, logger: BoundLogger
    ) -> Tuple[Optional[List[np.ndarray]], Optional[np.ndarray]]:
        """
        Resolves the `points` and `offset` of the `KMesh` from the `grid` and the `center`.

        Args:
            logger (BoundLogger): The logger to log messages.

        Returns:
            (Optional[List[pint.Quantity, pint.Quantity]]): The resolved `points` and `offset` of the `KMesh`.
        """
        points = None
        offset = None
        if self.center == 'Gamma-centered':
            grid_space = [np.linspace(0, 1, n) for n in self.grid]
            points = np.meshgrid(grid_space)
            offset = np.array([0, 0, 0])
        elif self.center == 'Monkhorst-Pack':
            try:
                points = monkhorst_pack(self.grid)
                offset = get_monkhorst_pack_size_and_offset(points)[-1]
            except ValueError:
                logger.warning(
                    'Could not resolve `KMesh.points` and `KMesh.offset` from `KMesh.grid`. ASE `monkhorst_pack` failed.'
                )
                return None  # this is a quick workaround: k_mesh.grid should be symmetry reduced
        return points, offset

    def get_k_line_density(
        self, reciprocal_lattice_vectors: pint.Quantity, logger: BoundLogger
    ) -> Optional[np.float64]:
        """
        Gets the k-line density of the `KMesh`. This quantity is used as a precision measure
        of the `KMesh` sampling.

        Args:
            reciprocal_lattice_vectors (pint.Quantity, [3, 3]): Reciprocal lattice vectors of the atomic cell.

        Returns:
            (np.float64): The k-line density of the `KMesh`.
        """
        if reciprocal_lattice_vectors is None:
            logger.error('No `reciprocal_lattice_vectors` input found.')
            return None
        if len(reciprocal_lattice_vectors) != 3 or len(self.grid) != 3:
            logger.error(
                'The `reciprocal_lattice_vectors` and the `grid` should have the same dimensionality.'
            )
            return None

        reciprocal_lattice_vectors = reciprocal_lattice_vectors.magnitude
        return min(
            [
                k_point / (np.linalg.norm(k_vector))
                for k_vector, k_point in zip(reciprocal_lattice_vectors, self.grid)
            ]
        )

    def resolve_k_line_density(
        self, model_systems: List[ModelSystem], logger: BoundLogger
    ) -> Optional[pint.Quantity]:
        """
        Resolves the `k_line_density` of the `KMesh` from the the list of `ModelSystem`.

        Args:
            model_systems (List[ModelSystem]): The list of `ModelSystem` sections.
            logger (BoundLogger): The logger to log messages.

        Returns:
            (Optional[pint.Quantity]): The resolved `k_line_density` of the `KMesh`.
        """
        for model_system in model_systems:
            # General checks to proceed with normalization
            if is_not_representative(model_system, logger):
                return None
            # TODO extend this for other dimensions (@ndaelman-hu)
            if model_system.type != 'bulk':
                logger.warning('`ModelSystem.type` is not describing a bulk system.')
                return None

            atomic_cell = model_system.cell
            if atomic_cell is None:
                logger.warning('`ModelSystem.cell` was not found.')
                return None

            # Set the `reciprocal_lattice_vectors` using ASE
            ase_atoms = atomic_cell[0].to_ase_atoms(logger)
            if self.reciprocal_lattice_vectors is None:
                self.reciprocal_lattice_vectors = (
                    2 * np.pi * ase_atoms.get_reciprocal_cell() / ureg.angstrom
                )

            # Resolve `k_line_density`
            if k_line_density := self.get_k_line_density(
                self.reciprocal_lattice_vectors, logger
            ):
                return k_line_density * ureg('m')
        return None

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        # If `grid` is not defined, we do not normalize the KMesh
        if self.grid is None:
            logger.warning('Could not find `KMesh.grid`.')
            return

        # Normalize k mesh from grid sampling
        if self.points is None and self.offset is None:
            self.points, self.offset = self.resolve_points_and_offset(logger)

        # Calculate k_line_density for data quality measures
        model_systems = self.m_xpath('m_parent.m_parent.model_system', dict=False)
        if self.k_line_density is None:
            self.k_line_density = self.resolve_k_line_density(model_systems, logger)


class QuasiparticlesFrequencyMesh(Mesh):
    """
    A base section used to specify the settings of a sampling mesh in the frequency real or imaginary space for quasiparticle calculations.
    """

    points = Quantity(
        type=np.complex128,
        shape=['n_points', 'dimensionality'],
        unit='joule',
        description="""
        List of all the points in the mesh in joules.
        """,
    )

    def __init__(self, m_def: Section = None, m_context: Context = None, **kwargs):
        super().__init__(m_def, m_context, **kwargs)
        # Set the name of the section
        self.name = self.m_def.name

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class SelfConsistency(NumericalSettings):
    """
    A base section used to define the convergence settings of self-consistent field (SCF) calculation.
    It determines the condictions for `is_converged` in properties `Outputs` (see outputs.py). The convergence
    criteria covered are:

        1. The number of iterations is smaller than or equal to `n_max_iterations`.

    and one of the following:

        2a. The total energy change between two subsequent self-consistent iterations is below
        `threshold_energy_change`.
        2b. The charge density change between two subsequent self-consistent iterations is below
        `threshold_charge_density_change`.
    """

    # TODO add examples or MEnum?
    scf_minimization_algorithm = Quantity(
        type=str,
        description="""
        Specifies the algorithm used for self consistency minimization.
        """,
    )

    n_max_iterations = Quantity(
        type=np.int32,
        description="""
        Specifies the maximum number of allowed self-consistent iterations. The simulation `is_converged`
        if the number of iterations is not larger or equal than this quantity.
        """,
    )

    # ? define class `Tolerance` for the different Scf tolerances types?
    threshold_energy_change = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        Specifies the threshold for the total energy change between two subsequent self-consistent iterations.
        The simulation `is_converged` if the total energy change between two SCF cycles is below
        this threshold.
        """,
    )

    threshold_charge_density_change = Quantity(
        type=np.float64,
        description="""
        Specifies the threshold for the average charge density change between two subsequent
        self-consistent iterations. The simulation `is_converged` if the charge density change
        between two SCF cycles is below this threshold.
        """,
    )

    def __init__(self, m_def: Section = None, m_context: Context = None, **kwargs):
        super().__init__(m_def, m_context, **kwargs)
        # Set the name of the section
        self.name = self.m_def.name

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class BasisSet(NumericalSettings):
    """"""

    # TODO work on this base section (@ndaelman-hu)

    def __init__(self, m_def: Section = None, m_context: Context = None, **kwargs):
        super().__init__(m_def, m_context, **kwargs)
        # Set the name of the section
        self.name = self.m_def.name

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)
