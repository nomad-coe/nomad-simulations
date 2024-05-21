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
import itertools
from structlog.stdlib import BoundLogger
from typing import Optional, List, Tuple, Union
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

from nomad_simulations.model_system import ModelSystem
from nomad_simulations.utils import is_not_representative


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


class Mesh(ArchiveSection):
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


class KMesh(Mesh):
    """
    A base section used to specify the settings of a sampling mesh in reciprocal space.
    """

    label = Quantity(
        type=MEnum('k-mesh', 'q-mesh'),
        default='k-mesh',
        description="""
        Label used to identify the `KMesh` with the reciprocal vector used. In linear response, `k` is used for
        refering to the wave-vector of electrons, while `q` is used for the scattering effect of the Coulomb potential.
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
        Full list of the mesh points without any symmetry operations. In the presence of symmetry operations, this quantity is a
        larger list than `points` (as it will contain all the points in the Brillouin zone).
        """,
    )

    high_symmetry_points = Quantity(
        type=JSON,
        description="""
        Dictionary containing the high-symmetry points and their points in terms of `reciprocal_lattice_vectors`.
        E.g., in a cubic lattice:
            high_symmetry_points = {
                'Gamma1': [0, 0, 0],
                'X': [0.5, 0, 0],
                ...
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

    # TODO add extraction of `high_symmetry_points` using BandStructureNormalizer idea (left for later when defining outputs.py)

    def _check_reciprocal_lattice_vectors(
        self, reciprocal_lattice_vectors: Optional[pint.Quantity], logger: BoundLogger
    ) -> bool:
        """
        Check if the `reciprocal_lattice_vectors` exist and if they have the same dimensionality as `grid`.

        Args:
            reciprocal_lattice_vectors (Optional[pint.Quantity]): The reciprocal lattice vectors of the atomic cell.
            logger (BoundLogger): The logger to log messages.

        Returns:
            (bool): True if the `reciprocal_lattice_vectors` exist and have the same dimensionality as `grid`, False otherwise.
        """
        if reciprocal_lattice_vectors is None:
            logger.warning(
                'Could not find `reciprocal_lattice_vectors` from parent `KSpace`.'
            )
            return False
        if self.grid is None:
            logger.warning('Could not find `KMesh.grid`.')
            return False
        if len(reciprocal_lattice_vectors) != 3 or len(self.grid) != 3:
            logger.warning(
                'The `reciprocal_lattice_vectors` and the `grid` should have the same dimensionality.'
            )
            return False
        return True

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
        if self.grid is None:
            logger.warning('Could not find `KMesh.grid`.')
            return None, None

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
                # this is a quick workaround: k_mesh.grid should be symmetry reduced
                return None, None
        return points, offset

    def get_k_line_density(
        self, reciprocal_lattice_vectors: Optional[pint.Quantity], logger: BoundLogger
    ) -> Optional[np.float64]:
        """
        Gets the k-line density of the `KMesh`. This quantity is used as a precision measure
        of the `KMesh` sampling.

        Args:
            reciprocal_lattice_vectors (pint.Quantity, [3, 3]): Reciprocal lattice vectors of the atomic cell.

        Returns:
            (np.float64): The k-line density of the `KMesh`.
        """
        # Initial check
        if not self._check_reciprocal_lattice_vectors(
            reciprocal_lattice_vectors, logger
        ):
            return None

        rlv = reciprocal_lattice_vectors.magnitude
        k_line_density = min(
            [
                k_point / (np.linalg.norm(k_vector))
                for k_vector, k_point in zip(rlv, self.grid)
            ]
        )
        return k_line_density / reciprocal_lattice_vectors.u

    def resolve_k_line_density(
        self,
        model_systems: List[ModelSystem],
        reciprocal_lattice_vectors: pint.Quantity,
        logger: BoundLogger,
    ) -> Optional[pint.Quantity]:
        """
        Resolves the `k_line_density` of the `KMesh` from the the list of `ModelSystem`.

        Args:
            model_systems (List[ModelSystem]): The list of `ModelSystem` sections.
            logger (BoundLogger): The logger to log messages.

        Returns:
            (Optional[pint.Quantity]): The resolved `k_line_density` of the `KMesh`.
        """
        # Initial check
        if not self._check_reciprocal_lattice_vectors(
            reciprocal_lattice_vectors, logger
        ):
            return None

        for model_system in model_systems:
            # General checks to proceed with normalization
            if is_not_representative(model_system, logger):
                return None
            # TODO extend this for other dimensions (@ndaelman-hu)
            if model_system.type != 'bulk':
                logger.warning('`ModelSystem.type` is not describing a bulk system.')
                return None

            # Resolve `k_line_density`
            if k_line_density := self.get_k_line_density(
                reciprocal_lattice_vectors, logger
            ):
                return k_line_density
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
        model_systems = self.m_xpath(
            'm_parent.m_parent.m_parent.model_system', dict=False
        )
        reciprocal_lattice_vectors = self.m_xpath(
            'm_parent.reciprocal_lattice_vectors', dict=False
        )
        if self.k_line_density is None:
            self.k_line_density = self.resolve_k_line_density(
                model_systems, reciprocal_lattice_vectors, logger
            )


class KLinePath(ArchiveSection):
    """
    A base section used to define the settings of a k-line path within a multidimensional mesh.
    """

    high_symmetry_path = Quantity(
        type=JSON,
        description="""
        Dictionary containing the high-symmetry points (in units of the `reciprocal_lattice_vectors`) followed in
        the k-line path. E.g., in a cubic lattice:
            high_symmetry_path = {
                'Gamma': [0, 0, 0],
                'X': [0.5, 0, 0],
                'Y': [0, 0.5, 0],
            }
        """,
    )

    n_line_points = Quantity(
        type=np.int32,
        description="""
        Number of points in the k-line path.
        """,
    )

    points = Quantity(
        type=np.float64,
        shape=['n_line_points', 3],
        description="""
        List of all the points in the k-line path in units of the `reciprocal_lattice_vectors`.
        """,
    )

    def get_high_symmetry_points_norm(
        self,
        reciprocal_lattice_vectors: Optional[pint.Quantity],
    ) -> Optional[dict]:
        """
        Get the high symmetry points norms from the dictionary of vectors in units of the `reciprocal_lattice_vectors`. This
        function is useful when matching lists of points passed as norms to the high symmetry points in order to resolve
        `KLinePath.points`

        Args:
            reciprocal_lattice_vectors (Optional[np.ndarray]): The reciprocal lattice vectors of the atomic cell.

        Returns:
            (Optional[dict]): The high symmetry points norms.
        """
        # Checking if `reciprocal_lattice_vectors` is defined and taking its magnitude to operate
        if reciprocal_lattice_vectors is None:
            return None
        rlv = reciprocal_lattice_vectors.magnitude

        # initializing the norms dictionary
        high_symmetry_points_norms = {
            key: 0.0 * reciprocal_lattice_vectors.u
            for key in self.high_symmetry_path.keys()
        }
        # initializing the first point
        prev_value_norm = 0.0 * reciprocal_lattice_vectors.u
        prev_value_rlv = np.array([0, 0, 0])
        for i, (key, value) in enumerate(self.high_symmetry_path.items()):
            if i == 0:
                continue
            value_rlv = value @ rlv
            value_tot_rlv = value_rlv - prev_value_rlv
            value_norm = (
                np.linalg.norm(value_tot_rlv) * reciprocal_lattice_vectors.u
                + prev_value_norm
            )
            high_symmetry_points_norms[key] = value_norm

            # accumulate value vector and norm
            prev_value_rlv = value_rlv
            prev_value_norm = value_norm
        return high_symmetry_points_norms

    def resolve_points(
        self,
        points_norm: Union[np.ndarray, List[float]],
        reciprocal_lattice_vectors: Optional[np.ndarray],
        logger: BoundLogger,
    ) -> None:
        """
        Resolves the `points` of the `KLinePath` from the `points_norm` and the `reciprocal_lattice_vectors`. This is useful
        when a list of points norms and the dictionary of high symmetry points are passed to resolve the `KLinePath.points`.

        Args:
            points_norm (List[float]): List of points norms in the k-line path.
            reciprocal_lattice_vectors (Optional[np.ndarray]): The reciprocal lattice vectors of the atomic cell.
            logger (BoundLogger): The logger to log messages.
        """
        # General checks for quantities
        if self.high_symmetry_path is None:
            logger.warning('Could not resolve `KLinePath.high_symmetry_path`.')
            return None
        if reciprocal_lattice_vectors is None:
            logger.warning(
                'The `reciprocal_lattice_vectors` are not passed as an input.'
            )
            return None
        # Check if `points_norm` is a list and convert it to a numpy array
        if isinstance(points_norm, list):
            points_norm = np.array(points_norm)

        # Define `n_line_points`
        if self.n_line_points is not None and len(points_norm) != self.n_line_points:
            logger.info(
                'The length of the `points` and the stored `n_line_points` do not coincide. We will overwrite `n_line_points` with the new length of `points`.'
            )
        self.n_line_points = len(points_norm)

        # Calculate the total norm of the path in order to find the closest indices in the list of `points_norm`
        high_symmetry_points_norms = self.get_high_symmetry_points_norm(
            reciprocal_lattice_vectors
        )
        closest_indices = {}
        for key, norm in high_symmetry_points_norms.items():
            closest_idx = (np.abs(points_norm - norm.magnitude)).argmin()
            closest_indices[key] = closest_idx

        # Append the data in the new `points` in units of the `reciprocal_lattice_vectors`
        points = []
        for i, (key, value) in enumerate(self.high_symmetry_path.items()):
            if i == 0:
                prev_value = value
                prev_index = closest_indices[key]
                continue
            elif i == len(self.high_symmetry_path) - 1:
                points.append(
                    np.linspace(
                        prev_value, value, num=closest_indices[key] - prev_index + 1
                    )
                )
            else:
                # pop the last element as it appears repeated in the next segment
                points.append(
                    np.linspace(
                        prev_value, value, num=closest_indices[key] - prev_index + 1
                    )[:-1]
                )
            prev_value = value
            prev_index = closest_indices[key]
        new_points = list(itertools.chain(*points))
        # And store this information in the `points` quantity
        if self.points is not None:
            logger.info('Overwriting `KLinePath.points` with the resolved points.')
        self.points = new_points

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class KSpace(NumericalSettings):
    """
    A base section used to specify the settings of the k-space. This section contains two main sub-sections,
    depending on the k-space sampling: `k_mesh` or `k_line_path`.
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

    k_mesh = SubSection(sub_section=KMesh.m_def, repeats=True)

    k_line_path = SubSection(sub_section=KLinePath.m_def)

    def __init__(self, m_def: Section = None, m_context: Context = None, **kwargs):
        super().__init__(m_def, m_context, **kwargs)
        # Set the name of the section
        self.name = self.m_def.name

    def resolve_reciprocal_lattice_vectors(
        self, model_systems: List[ModelSystem], logger: BoundLogger
    ) -> Optional[pint.Quantity]:
        """
        Resolve the `reciprocal_lattice_vectors` of the `KSpace` from the representative `ModelSystem` section.

        Args:
            model_systems (List[ModelSystem]): The list of `ModelSystem` sections.
            logger (BoundLogger): The logger to log messages.

        Returns:
            (Optional[pint.Quantity]): The resolved `reciprocal_lattice_vectors` of the `KSpace`.
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
            return 2 * np.pi * ase_atoms.get_reciprocal_cell() / ureg.angstrom
        return None

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        # Resolve `reciprocal_lattice_vectors` from the `ModelSystem` ASE object
        model_systems = self.m_xpath('m_parent.m_parent.model_system', dict=False)
        if self.reciprocal_lattice_vectors is None:
            self.reciprocal_lattice_vectors = self.resolve_reciprocal_lattice_vectors(
                model_systems, logger
            )


class SelfConsistency(NumericalSettings):
    """
    A base section used to define the convergence settings of self-consistent field (SCF) calculation.
    It determines the condictions for `is_scf_converged` in `SCFOutputs` (see outputs.py). The convergence
    criteria covered are:

        1. The number of iterations is smaller than or equal to `n_max_iterations`.
        2. The total change between two subsequent self-consistent iterations for an output property is below
        `threshold_change`.
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
        Specifies the maximum number of allowed self-consistent iterations. The simulation `is_scf_converged`
        if the number of iterations is not larger or equal than this quantity.
        """,
    )

    threshold_change = Quantity(
        type=np.float64,
        description="""
        Specifies the threshold for the change between two subsequent self-consistent iterations on
        a given output property. The simulation `is_scf_converged` if this total change is below
        this threshold.
        """,
    )

    threshold_change_unit = Quantity(
        type=str,
        description="""
        Unit using the pint UnitRegistry() notation for the `threshold_change`.
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
