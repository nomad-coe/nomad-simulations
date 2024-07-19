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

from itertools import accumulate, chain, tee
from typing import TYPE_CHECKING, Optional, Union

from nomad_simulations.schema_packages.model_method import BaseModelMethod
import numpy as np
import pint
from ase.dft.kpoints import get_monkhorst_pack_size_and_offset, monkhorst_pack
from nomad.datamodel.data import ArchiveSection
from nomad.datamodel.metainfo.annotations import ELNAnnotation
from nomad.metainfo import JSON, MEnum, Quantity, SubSection
from nomad.units import ureg

from nomad_simulations.schema_packages.atoms_state import AtomsState

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from nomad.metainfo import Context, Section
    from structlog.stdlib import BoundLogger

from nomad_simulations.schema_packages.model_system import ModelSystem
from nomad_simulations.schema_packages.utils import is_not_representative


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

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class Smearing(NumericalSettings):
    """
    Section specifying the smearing of the occupation numbers to
    either simulate temperature effects or improve SCF convergence.
    """

    name = Quantity(
        type=MEnum('Fermi-Dirac', 'Gaussian', 'Methfessel-Paxton'),
        description="""
        Smearing routine employed.
        """,
    )


class Mesh(ArchiveSection):
    """
    A base section used to specify the settings of a sampling mesh.
    It supports uniformly-spaced meshes and symmetry-reduced representations.
    """

    spacing = Quantity(
        type=MEnum('Equidistant', 'Logarithmic', 'Tan'),
        shape=['dimensionality'],
        description="""
        Identifier for the spacing of the Mesh. Defaults to 'Equidistant' if not defined. It can take the values:

        | Name      | Description                      |
        | --------- | -------------------------------- |
        | `'Equidistant'`  | Equidistant grid (also known as 'Newton-Cotes') |
        | `'Logarithmic'`  | log distance grid |
        | `'Tan'`  | Non-uniform tan mesh for grids. More dense at low abs values of the points, while less dense for higher values |
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
    )  # ! @JosePizarro3 I think that this is separate from the spacing

    n_points = Quantity(
        type=np.int32,
        description="""
        Number of points in the mesh.
        """,
    )

    dimensionality = Quantity(
        type=MEnum(1, 2, 3),
        default=3,
        description="""
        Dimensionality of the mesh: 1, 2, or 3. Defaults to 3.
        """,
    )

    grid = Quantity(
        type=np.int32,
        shape=['dimensionality'],
        description="""
        Amount of mesh point sampling along each axis. See `type` for the axes definition.
        """,
    )  # ? @JosePizzaro3: should the mesh also contain its boundary information

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
        a symmetry operation that was applied to the `Mesh`. This quantity is equivalent to `weights`:

            multiplicities = n_points * weights
        """,
    )

    weights = Quantity(
        type=np.float64,
        shape=['n_points'],
        description="""
        Weight of each point. A value smaller than 1, typically indicates a symmetry operation that was
        applied to the mesh. This quantity is equivalent to `multiplicities`:

            weights = multiplicities / n_points
        """,
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class KSpaceFunctionalities:
    """
    A functionality class useful for defining methods shared between `KSpace`, `KMesh`, and `KLinePath`.
    """

    def validate_reciprocal_lattice_vectors(
        self,
        reciprocal_lattice_vectors: Optional[pint.Quantity],
        logger: 'BoundLogger',
        check_grid: Optional[bool] = False,
        grid: Optional[list[int]] = [],
    ) -> bool:
        """
        Validate the `reciprocal_lattice_vectors` by checking if they exist and if they have the same dimensionality as `grid`.

        Args:
            reciprocal_lattice_vectors (Optional[pint.Quantity]): The reciprocal lattice vectors of the atomic cell.
            logger (BoundLogger): The logger to log messages.
            check_grid (bool, optional): Flag to check the `grid` is set to True. Defaults to False.
            grid (Optional[list[int]], optional): The grid of the `KMesh`. Defaults to [].

        Returns:
            (bool): True if the `reciprocal_lattice_vectors` exist. If `check_grid_too` is set to True, it also checks if the
            `reciprocal_lattice_vectors` and the `grid` have the same dimensionality. False otherwise.
        """
        if reciprocal_lattice_vectors is None:
            logger.warning('Could not find `reciprocal_lattice_vectors`.')
            return False
        # Only checking the `reciprocal_lattice_vectors`
        if not check_grid:
            return True

        # Checking the `grid` too
        if grid is None:
            logger.warning('Could not find `grid`.')
            return False
        if len(reciprocal_lattice_vectors) != 3 or len(grid) != 3:
            logger.warning(
                'The `reciprocal_lattice_vectors` and the `grid` should have the same dimensionality.'
            )
            return False
        return True

    def resolve_high_symmetry_points(
        self,
        model_systems: list[ModelSystem],
        logger: 'BoundLogger',
        eps: float = 3e-3,
    ) -> Optional[dict]:
        """
        Resolves the `high_symmetry_points` from the list of `ModelSystem`. This method relies on using the `ModelSystem`
        information in the sub-sections `Symmetry` and `AtomicCell`, and uses the ASE package to extract the
        special (high symmetry) points information.

        Args:
            model_systems (list[ModelSystem]): The list of `ModelSystem` sections.
            logger (BoundLogger): The logger to log messages.
            eps (float, optional): Tolerance factor to define the `lattice` ASE object. Defaults to 3e-3.

        Returns:
            (Optional[dict]): The resolved `high_symmetry_points`.
        """
        # Extracting `bravais_lattice` from `ModelSystem.symmetry` section and `ASE.cell` from `ModelSystem.cell`
        lattice = None
        for model_system in model_systems:
            # General checks to proceed with normalization
            if is_not_representative(model_system=model_system, logger=logger):
                continue
            if model_system.symmetry is None:
                logger.warning('Could not find `ModelSystem.symmetry`.')
                continue
            bravais_lattice = [symm.bravais_lattice for symm in model_system.symmetry]
            if len(bravais_lattice) != 1:
                logger.warning(
                    'Could not uniquely determine `bravais_lattice` from `ModelSystem.symmetry`.'
                )
                continue
            bravais_lattice = bravais_lattice[0]

            if model_system.cell is None:
                logger.warning('Could not find `ModelSystem.cell`.')
                continue
            prim_atomic_cell = None
            for atomic_cell in model_system.cell:
                if atomic_cell.type == 'primitive':
                    prim_atomic_cell = atomic_cell
                    break
            if prim_atomic_cell is None:
                logger.warning(
                    'Could not find the primitive `AtomicCell` under `ModelSystem.cell`.'
                )
                continue
            # function defined in AtomicCell
            atoms = prim_atomic_cell.to_ase_atoms(logger=logger)
            cell = atoms.get_cell()
            lattice = cell.get_bravais_lattice(eps=eps)
            break  # only cover the first representative `ModelSystem`

        # Checking if `bravais_lattice` and `lattice` are defined
        if lattice is None:
            logger.warning(
                'Could not resolve `bravais_lattice` and `lattice` ASE object from the `ModelSystem`.'
            )
            return None

        # Non-conventional ordering testing for certain lattices:
        if bravais_lattice in ['oP', 'oF', 'oI', 'oS']:
            a, b, c = lattice.a, lattice.b, lattice.c
            assert a < b
            if bravais_lattice != 'oS':
                assert b < c
        elif bravais_lattice in ['mP', 'mS']:
            a, b, c = lattice.a, lattice.b, lattice.c
            alpha = lattice.alpha * np.pi / 180
            assert a <= c and b <= c  # ordering of the conventional lattice
            assert alpha < np.pi / 2

        # Extracting the `high_symmetry_points` from the `lattice` object
        special_points = lattice.get_special_points()
        if special_points is None:
            logger.warning(
                'Could not find `lattice.get_special_points()` from the ASE package.'
            )
            return None
        high_symmetry_points = {}
        for key, value in lattice.get_special_points().items():
            if key == 'G':
                key = 'Gamma'
            if bravais_lattice == 'tI':
                if key == 'S':
                    key = 'Sigma'
                elif key == 'S1':
                    key = 'Sigma1'
            high_symmetry_points[key] = list(value)
        return high_symmetry_points


class KMesh(Mesh):
    """
    A base section used to specify the settings of a sampling mesh in reciprocal space. The `points` and other
    k-space quantities are defined in units of the reciprocal lattice vectors, so that to obtain their Cartesian coordinates
    value, one should multiply them by the reciprocal lattice vectors (`points_cartesian = points @ reciprocal_lattice_vectors`).
    """

    label = Quantity(
        type=MEnum('k-mesh', 'q-mesh'),  # ? add g-mesh for LAPW
        default='k-mesh',
        description="""
        Label used to identify the `KMesh` with the reciprocal vector used. In linear response, `k` is used for
        refering to the wave-vector of electrons, while `q` is used for the scattering effect of the Coulomb potential.
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
        Full list of the mesh points without any symmetry operations in units of the `reciprocal_lattice_vectors`. In the
        presence of symmetry operations, this quantity is a larger list than `points` (as it will contain all the points
        in the Brillouin zone).
        """,
    )

    high_symmetry_points = Quantity(
        type=JSON,
        description="""
        Dictionary containing the high-symmetry point labels and their values in units of `reciprocal_lattice_vectors`.
        E.g., in a cubic lattice:
            high_symmetry_points = {
                'Gamma': [0, 0, 0],
                'X': [0.5, 0, 0],
                'Y': [0, 0.5, 0],
                ...
            ]
        """,
    )

    k_line_density = Quantity(
        type=np.float64,
        unit='m',
        description="""
        Amount of sampled k-points per unit reciprocal length along each axis. Contains the least precise density out of all axes.
        Should only be compared between calulations of similar dimensionality.
        """,
    )

    # TODO add extraction of `high_symmetry_points` using BandStructureNormalizer idea (left for later when defining outputs.py)

    def resolve_points_and_offset(
        self, logger: 'BoundLogger'
    ) -> tuple[Optional[list[np.ndarray]], Optional[np.ndarray]]:
        """
        Resolves the `points` and `offset` of the `KMesh` from the `grid` and the `center`.

        Args:
            logger (BoundLogger): The logger to log messages.

        Returns:
            (tuple[Optional[list[np.ndarray]], Optional[np.ndarray]]): The resolved `points` and `offset` of the `KMesh`.
        """
        if self.grid is None:
            logger.warning('Could not find `KMesh.grid`.')
            return None, None

        points = None
        offset = None
        if self.center == 'Gamma-centered':  # ! fix this (@ndaelman-hu)
            grid_space = [np.linspace(0, 1, n) for n in self.grid]
            points = np.meshgrid(grid_space)
            offset = np.array([0, 0, 0])
        elif self.center == 'Monkhorst-Pack':
            try:
                points = monkhorst_pack(size=self.grid)
                offset = get_monkhorst_pack_size_and_offset(kpts=points)[-1]
            except ValueError:
                logger.warning(
                    'Could not resolve `KMesh.points` and `KMesh.offset` from `KMesh.grid`. ASE `monkhorst_pack` failed.'
                )
                # this is a quick workaround: k_mesh.grid should be symmetry reduced
                return None, None
        return points, offset

    def get_k_line_density(
        self, reciprocal_lattice_vectors: Optional[pint.Quantity], logger: 'BoundLogger'
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
        if not KSpaceFunctionalities().validate_reciprocal_lattice_vectors(
            reciprocal_lattice_vectors=reciprocal_lattice_vectors,
            logger=logger,
            check_grid=True,
            grid=self.grid,
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
        model_systems: list[ModelSystem],
        reciprocal_lattice_vectors: pint.Quantity,
        logger: 'BoundLogger',
    ) -> Optional[pint.Quantity]:
        """
        Resolves the `k_line_density` of the `KMesh` from the the list of `ModelSystem`.

        Args:
            model_systems (list[ModelSystem]): The list of `ModelSystem` sections.
            logger (BoundLogger): The logger to log messages.

        Returns:
            (Optional[pint.Quantity]): The resolved `k_line_density` of the `KMesh`.
        """
        # Initial check
        if not KSpaceFunctionalities().validate_reciprocal_lattice_vectors(
            reciprocal_lattice_vectors=reciprocal_lattice_vectors,
            logger=logger,
            check_grid=True,
            grid=self.grid,
        ):
            return None

        for model_system in model_systems:
            # General checks to proceed with normalization
            if is_not_representative(model_system=model_system, logger=logger):
                continue
            # TODO extend this for other dimensions (@ndaelman-hu)
            if model_system.type != 'bulk':
                logger.warning('`ModelSystem.type` is not describing a bulk system.')
                continue

            # Resolve `k_line_density`
            if k_line_density := self.get_k_line_density(
                reciprocal_lattice_vectors=reciprocal_lattice_vectors, logger=logger
            ):
                return k_line_density
        return None

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

        # If `grid` is not defined, we do not normalize the KMesh
        if self.grid is None:
            logger.warning('Could not find `KMesh.grid`.')
            return

        # Normalize k mesh from grid sampling
        if self.points is None and self.offset is None:
            self.points, self.offset = self.resolve_points_and_offset(logger=logger)

        # Calculate k_line_density for data quality measures
        model_systems = self.m_xpath(
            'm_parent.m_parent.m_parent.model_system', dict=False
        )
        reciprocal_lattice_vectors = self.m_xpath(
            'm_parent.reciprocal_lattice_vectors', dict=False
        )
        if self.k_line_density is None:
            self.k_line_density = self.resolve_k_line_density(
                model_systems=model_systems,
                reciprocal_lattice_vectors=reciprocal_lattice_vectors,
                logger=logger,
            )

        # Resolve `high_symmetry_points`
        if self.high_symmetry_points is None:
            self.high_symmetry_points = (
                KSpaceFunctionalities().resolve_high_symmetry_points(
                    model_systems=model_systems, logger=logger
                )
            )


class KLinePath(ArchiveSection):
    """
    A base section used to define the settings of a k-line path within a multidimensional mesh. The `points` and other
    k-space quantities are defined in units of the reciprocal lattice vectors, so that to obtain their Cartesian coordinates
    value, one should multiply them by the reciprocal lattice vectors (`points_cartesian = points @ reciprocal_lattice_vectors`).
    """

    high_symmetry_path_names = Quantity(
        type=str,
        shape=['*'],
        description="""
        List of the high-symmetry path names followed in the k-line path. This quantity is directly coupled with `high_symmetry_path_value`.
        E.g., in a cubic lattice: `high_symmetry_path_names = ['Gamma', 'X', 'Y', 'Gamma']`.
        """,
    )

    high_symmetry_path_values = Quantity(
        type=np.float64,
        shape=['*', 3],
        description="""
        List of the high-symmetry path values in units of the `reciprocal_lattice_vectors` in the k-line path. This quantity is directly
        coupled with `high_symmetry_path_names`. E.g., in a cubic lattice: `high_symmetry_path_value = [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0]]`.
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

    def resolve_high_symmetry_path_values(
        self,
        model_systems: list[ModelSystem],
        reciprocal_lattice_vectors: pint.Quantity,
        logger: 'BoundLogger',
    ) -> Optional[list[float]]:
        """
        Resolves the `high_symmetry_path_values` of the `KLinePath` from the `high_symmetry_path_names`.

        Args:
            model_systems (list[ModelSystem]): The list of `ModelSystem` sections.
            reciprocal_lattice_vectors (pint.Quantity): The reciprocal lattice vectors of the atomic cell.
            logger (BoundLogger): The logger to log messages.

        Returns:
            (Optional[list[float]]): The resolved `high_symmetry_path_values`.
        """
        # Initial check on the `reciprocal_lattice_vectors`
        if not KSpaceFunctionalities().validate_reciprocal_lattice_vectors(
            reciprocal_lattice_vectors=reciprocal_lattice_vectors, logger=logger
        ):
            return []

        # Resolving the dictionary containing the `high_symmetry_points` for the given ModelSystem symmetry
        high_symmetry_points = KSpaceFunctionalities().resolve_high_symmetry_points(
            model_systems=model_systems, logger=logger
        )
        if high_symmetry_points is None:
            return []

        # Appending into a list which is stored in the `high_symmetry_path_values`. There is a check in the `normalize()`
        # function to ensure that the length of the `high_symmetry_path_names` and `high_symmetry_path_values` coincide.
        if self.high_symmetry_path_names is None:
            return []
        high_symmetry_path_values = [
            high_symmetry_points[name]
            for name in self.high_symmetry_path_names
            if name in high_symmetry_points.keys()
        ]
        return high_symmetry_path_values

    def validate_high_symmetry_path(self, logger: 'BoundLogger') -> bool:
        """
        Validate `high_symmetry_path_names` and `high_symmetry_path_values` by checking if they are defined and have the same length.

        Args:
            logger (BoundLogger): The logger to log messages.

        Returns:
            (bool): True if the `high_symmetry_path_names` and `high_symmetry_path_values` are defined and have the same length, False otherwise.
        """
        if (
            self.high_symmetry_path_names is None
            or self.high_symmetry_path_values is None
        ) or (
            len(self.high_symmetry_path_names) == 0
            or len(self.high_symmetry_path_values) == 0
        ):
            logger.warning(
                'Could not find `KLinePath.high_symmetry_path_names` or `KLinePath.high_symmetry_path_values`.'
            )
            return False
        if len(self.high_symmetry_path_names) != len(self.high_symmetry_path_values):
            logger.warning(
                'The length of `KLinePath.high_symmetry_path_names` and `KLinePath.high_symmetry_path_values` should coincide.'
            )
            return False
        return True

    def get_high_symmetry_path_norms(
        self,
        reciprocal_lattice_vectors: Optional[pint.Quantity],
        logger: 'BoundLogger',
    ) -> Optional[list[pint.Quantity]]:
        """
        Get the high symmetry path points norms from the list of dictionaries of vectors in units of the `reciprocal_lattice_vectors`.
        The norms are accummulated, such that the first high symmetry point in the path list has a norm of 0, while the others sum the
        previous norm. This function is useful when matching lists of points passed as norms to the high symmetry path in order to
        resolve `KLinePath.points`.

        Args:
            reciprocal_lattice_vectors (Optional[np.ndarray]): The reciprocal lattice vectors of the atomic cell.
            logger (BoundLogger): The logger to log messages.

        Returns:
            (Optional[list[pint.Quantity]]): The high symmetry points norms list, e.g. in a cubic lattice:
                `high_symmetry_path_value_norms = [0, 0.5, 0.5 + 1 / np.sqrt(2), 1 + 1 / np.sqrt(2)]`
        """
        # Checking the high symmetry path quantities
        if not self.validate_high_symmetry_path(logger=logger):
            return None
        # Checking if `reciprocal_lattice_vectors` is defined and taking its magnitude to operate
        if reciprocal_lattice_vectors is None:
            return None
        rlv = reciprocal_lattice_vectors.magnitude

        def calc_norms(
            value_rlv: np.ndarray, prev_value_rlv: np.ndarray
        ) -> pint.Quantity:
            value_tot_rlv = value_rlv - prev_value_rlv
            return np.linalg.norm(value_tot_rlv) * reciprocal_lattice_vectors.u

        # Compute `rlv` projections
        rlv_projections = list(
            map(lambda value: value @ rlv, self.high_symmetry_path_values)
        )

        # Create two iterators for the projections
        rlv_projections_1, rlv_projections_2 = tee(rlv_projections)

        # Skip the first element in the second iterator
        next(rlv_projections_2, None)

        # Calculate the norms using accumulate
        norms = accumulate(
            zip(rlv_projections_2, rlv_projections_1),
            lambda acc, value_pair: calc_norms(value_pair[0], value_pair[1]) + acc,
            initial=0.0 * reciprocal_lattice_vectors.u,
        )
        return list(norms)

    def resolve_points(
        self,
        points_norm: Union[np.ndarray, list[float]],
        reciprocal_lattice_vectors: Optional[np.ndarray],
        logger: 'BoundLogger',
    ) -> None:
        """
        Resolves the `points` of the `KLinePath` from the `points_norm` and the `reciprocal_lattice_vectors`. This is useful
        when a list of points norms and the list of dictionaries of the high symmetry path are passed to resolve the `KLinePath.points`.

        Args:
            points_norm (list[float]): List of points norms in the k-line path.
            reciprocal_lattice_vectors (Optional[np.ndarray]): The reciprocal lattice vectors of the atomic cell.
            logger (BoundLogger): The logger to log messages.
        """
        # General checks for quantities
        if not self.validate_high_symmetry_path(logger=logger):
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

        # Calculate the norms in the path and find the closest indices in points_norm to the high symmetry path norms
        high_symmetry_path_value_norms = self.get_high_symmetry_path_norms(
            reciprocal_lattice_vectors=reciprocal_lattice_vectors, logger=logger
        )
        closest_indices = list(
            map(
                lambda norm: (np.abs(points_norm - norm.magnitude)).argmin(),
                high_symmetry_path_value_norms,
            )
        )

        def linspace_segments(
            prev_value: np.ndarray, value: np.ndarray, num: int
        ) -> np.ndarray:
            return np.linspace(prev_value, value, num=num + 1)[:-1]

        # Generate point segments using `map` and `linspace_segments`
        points_segments = list(
            map(
                lambda i, value: linspace_segments(
                    self.high_symmetry_path_values[i - 1],
                    value,
                    closest_indices[i] - closest_indices[i - 1],
                )
                if i > 0
                else np.array([]),
                range(len(self.high_symmetry_path_values)),
                self.high_symmetry_path_values,
            )
        )
        # and handle the last segment to include all points
        points_segments[-1] = np.linspace(
            self.high_symmetry_path_values[-2],
            self.high_symmetry_path_values[-1],
            num=closest_indices[-1] - closest_indices[-2] + 1,
        )

        # Flatten the list of segments into a single list of points
        new_points = list(chain.from_iterable(points_segments))

        # And store this information in the `points` quantity
        if self.points is not None:
            logger.info('Overwriting `KLinePath.points` with the resolved points.')
        self.points = new_points

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

        # Resolves `high_symmetry_path_values` from `high_symmetry_path_names`
        model_systems = self.m_xpath(
            'm_parent.m_parent.m_parent.model_system', dict=False
        )
        reciprocal_lattice_vectors = self.m_xpath(
            'm_parent.reciprocal_lattice_vectors', dict=False
        )
        if (
            self.high_symmetry_path_values is None
            or len(self.high_symmetry_path_values) == 0
        ):
            self.high_symmetry_path_values = self.resolve_high_symmetry_path_values(
                model_systems=model_systems,
                reciprocal_lattice_vectors=reciprocal_lattice_vectors,
                logger=logger,
            )

        # If `high_symmetry_path` is not defined, we do not normalize the KLinePath
        if not self.validate_high_symmetry_path(logger=logger):
            return


class KSpace(NumericalSettings):
    """
    A base section used to specify the settings of the k-space. This section contains two main sub-sections,
    depending on the k-space sampling: `k_mesh` or `k_line_path`.
    """

    # ! This needs to be normalized first in order to extract the `reciprocal_lattice_vectors` from the `ModelSystem.cell` information
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

    def __init__(self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs):
        super().__init__(m_def, m_context, **kwargs)
        # Set the name of the section
        self.name = self.m_def.name

    def resolve_reciprocal_lattice_vectors(
        self, model_systems: list[ModelSystem], logger: 'BoundLogger'
    ) -> Optional[pint.Quantity]:
        """
        Resolve the `reciprocal_lattice_vectors` of the `KSpace` from the representative `ModelSystem` section.

        Args:
            model_systems (list[ModelSystem]): The list of `ModelSystem` sections.
            logger (BoundLogger): The logger to log messages.

        Returns:
            (Optional[pint.Quantity]): The resolved `reciprocal_lattice_vectors` of the `KSpace`.
        """
        for model_system in model_systems:
            # General checks to proceed with normalization
            if is_not_representative(model_system=model_system, logger=logger):
                continue

            # TODO extend this for other dimensions (@ndaelman-hu)
            if model_system.type is not None and model_system.type != 'bulk':
                logger.warning('`ModelSystem.type` is not describing a bulk system.')
                continue

            atomic_cell = model_system.cell
            if atomic_cell is None:
                logger.warning('`ModelSystem.cell` was not found.')
                continue

            # Set the `reciprocal_lattice_vectors` using ASE
            ase_atoms = atomic_cell[0].to_ase_atoms(logger=logger)
            return 2 * np.pi * ase_atoms.get_reciprocal_cell() / ureg.angstrom
        return None

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

        # Resolve `reciprocal_lattice_vectors` from the `ModelSystem` ASE object
        model_systems = self.m_xpath('m_parent.m_parent.model_system', dict=False)
        if self.reciprocal_lattice_vectors is None:
            self.reciprocal_lattice_vectors = self.resolve_reciprocal_lattice_vectors(
                model_systems=model_systems, logger=logger
            )


class SelfConsistency(NumericalSettings):
    """
    A base section used to define the convergence settings of self-consistent field (SCF) calculation.
    It determines the conditions for `is_scf_converged` in `SCFOutputs` (see outputs.py). The convergence
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

    def __init__(self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs):
        super().__init__(m_def, m_context, **kwargs)
        # Set the name of the section
        self.name = self.m_def.name

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class BasisSet(ArchiveSection):
    """A type section denoting a basis set component of a simulation.
    Should be used as a base section for more specialized sections.
    Allows for denoting the basis set's _scope_, i.e. to which entity it applies,
    e.g. atoms species, orbital type, Hamiltonian term.

    Examples include:
    - mesh-based basis sets, e.g. (projector-)(augmented) plane-wave basis sets
    - atom-centered basis sets, e.g. Gaussian-type basis sets, Slater-type orbitals, muffin-tin orbitals
    """

    species_scope = Quantity(
        type=AtomsState,
        shape=['*'],
        description="""
        Reference to the section `AtomsState` specifying the localization of the basis set.
        """,
        a_eln=ELNAnnotation(components='ReferenceEditQuantity'),
    )

    # TODO: add atom index-based instantiator for species if not present

    hamiltonian_scope = Quantity(
        type=BaseModelMethod,
        shape=['*'],
        description="""
        Reference to the section `BaseModelMethod` containing the information
        of the Hamiltonian term to which the basis set applies.
        """,
        a_eln=ELNAnnotation(components='ReferenceEditQuantity'),
    )

    # ? band_scope or orbital_scope: valence vs core

    def __init__(self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs):
        super().__init__(m_def, m_context, **kwargs)
        # Set the name of the section
        self.name = self.m_def.name


class PlaneWaveBasisSet(BasisSet, Mesh):
    """
    Basis set over a reciprocal mesh, where each point $k_n$ represents a planar-wave basis function $\frac{1}{\sqrt{\omega}} e^{i k_n r}$.
    Typically the grid itself is cartesian with only points within a designated sphere considered.
    The cutoff radius may be defined by a reciprocal length, or more commonly, the equivalent kinetic energy for a free particle.

    * D. J. Singh and L. Nordström, \"Why Planewaves\" in Planewaves, pseudopotentials, and the LAPW method, 2nd ed. New York, NY: Springer, 2006, pp. 24-26.
    """

    cutoff_energy = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        Cutoff energy for the plane-wave basis set. The simulation uses plane waves with energies below this cutoff.
        """,
    )

    def normalize(self, archive: EntryArchive, logger: BoundLogger) -> None:
        super(BasisSet, self).normalize(archive, logger)
        super(Mesh, self).normalize(archive, logger)


class AtomCenteredFunction(ArchiveSection):
    """
    Specifies a single function (term) in an atom-centered basis set.
    """

    pass

    # TODO: design system for writing basis functions like gaussian or slater orbitals


class AtomCenteredBasisSet(BasisSet):
    """
    Defines an atom-centered basis set.
    """

    functional_composition = SubSection(
        sub_section=AtomCenteredFunction.m_def, repeats=True
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)
        # self.name = self.m_def.name
        # TODO: set name based on basis functions
        # ? use naming BSE


class APWWavefunction(ArchiveSection):
    """Abstract base section for (S)(L)APW and local orbital component wavefunctions.
    It helps defines the interface with `APWLChannel`."""

    pass


class APWOrbital(APWWavefunction):
    """
    Implementation of `APWWavefunction` capturing the foundational (S)(L)APW basis sets, all of the form $\sum_{lm} \left[ \sum_o c_{lmo} \frac{\partial}{\partial r}u_l(r, \epsilon_l) \right] Y_lm$.
    The energy parameter $\epsilon_l$ is always considered fixed during diagonalization, opposed to the original APW formulation.
    This representation then has to match the plane-wave $k_n$ points within the muffin-tin sphere.

    * D. J. Singh and L. Nordström, \"INTRODUCTION TO THE LAPW METHOD,\" in Planewaves, pseudopotentials, and the LAPW method, 2nd ed. New York, NY: Springer, 2006, pp. 43-52.
    """

    type = Quantity(
        type=MEnum('apw', 'lapw', 'slapw', 'spherical_dirac'),
        description="""
        Type of augmentation contribution. Abbreviations stand for:
        | name | description | radial product |
        |------|-------------|----------------|
        | APW  | augmented plane wave with a frozen energy parameter | $A_{lm, k_n} u_l (r, E_l)$ |
        | LAPW | linearized augmented plane wave with an optimized energy parameter | $A_{lm, k_n} u_l (r, E_l) + B_{lm, k_n} \dot{u}_{lm} (r, E_l^')$ |
        | SLAPW | super linearized augmented plane wave | -- |
        | spherical Dirac | spherical Dirac basis set | -- |

        * http://susi.theochem.tuwien.ac.at/lapw/
        """,
    )

    energy_parameter = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        Reference energy parameter for the augmented plane wave (APW) basis set.
        Is used to set the energy parameter for each state.
        """,
    )  # TODO: add approximation formula from energy parameter n

    energy_parameter_n = Quantity(
        type=np.int32,
        description="""
        Reference number of radial nodes for the augmented plane wave (APW) basis set.
        This is used to derive the `energy_parameter`.
        """,
    )

    energy_status = Quantity(
        type=MEnum('fixed', 'pre-optimization', 'post-optimization'),
        description="""
        Allow the code to optimize the initial energy parameter.
        """,
    )

    differential_order = Quantity(
        type=np.int32,
        description="""
        Derivative order of the radial wavefunction term.
        """,
    )  # TODO: add check non-negative # ? to remove

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)
        if self.differential_order < 0:
            logger.error('`APWLOrbital.differential_order` must be non-negative.')


class APWLocalOrbital(APWWavefunction):
    """
    Implementation of `APWWavefunction` capturing a local orbital extending a foundational APW basis set.
    Local orbitals allow for flexible additions to an `APWOrbital` specification.
    They may be included to describe semi-core states, virtual states, ghost bands, or improve overall convergence.

    * D. J. Singh and L. Nordström, \"Role of the Linearization Energies,\" in Planewaves, pseudopotentials, and the LAPW method, 2nd ed. New York, NY: Springer, 2006, pp. 49-52.
    """

    type = Quantity(
        type=MEnum('lo', 'LO', 'custom'),
        description="""
        Type of augmentation contribution. Abbreviations stand for:
        | name | description | radial product |
        |------|-------------|----------------|
        | lo   | 2-parameter local orbital | $A_l u_l (r, E_l) + B_l \dot{u}_l (r, E_l^')$ |
        | LO   | 3-parameter local orbital | $A_l u_l (r, E_l) + B_l \dot{u}_l (r, E_l^') + C_l \dot{u}_l (r, E_l^{''})$ |
        | custom | local orbital of a different formula |

        * http://susi.theochem.tuwien.ac.at/lapw/
        """,
    )

    n_terms = Quantity(
        type=np.int32,
        description="""
        Number of terms in the local orbital.
        """,
    )

    energy_parameters = Quantity(
        type=np.float64,
        shape=['n_terms'],
        unit='joule',
        description="""
        Reference energy parameter for the augmented plane wave (APW) basis set.
        Is used to set the energy parameter for each state.
        """,
    )  # TODO: add approximation formula from energy parameter n

    energy_parameters_n = Quantity(
        type=np.int32,
        shape=['n_terms'],
        description="""
        Reference number of radial nodes for the augmented plane wave (APW) basis set.
        This is used to derive the `energy_parameter`.
        """,
    )

    energy_statuses = Quantity(
        type=MEnum('fixed', 'pre-optimization', 'post-optimization'),
        shape=['n_terms'],
        description="""
        Allow the code to optimize the initial energy parameter.
        """,
    )

    differential_orders = Quantity(
        type=np.int32,
        shape=['n_terms'],
        description="""
        Derivative order of the radial wavefunction term.
        """,
    )  # TODO: add check non-negative # ? to remove

    boundary_orders = Quantity(
        type=np.int32,
        shape=['n_terms'],
        description="""
        Differential order to which the radial wavefunction is matched at the boundary.
        """,
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)
        if np.any(np.isneginf(self.differential_orders)):
            logger.error('`APWLOrbital.differential_order` must be non-negative.')
        if np.any(np.isneginf(self.boundary_orders)):
            logger.error('`APWLOrbital.boundary_orders` must be non-negative.')


class APWLChannel(BasisSet):
    """
    Collection of all (S)(L)APW and local orbital components that contribute
    to a single $l$-channel. $l$ here stands for the angular momentum parameter
    in the Laplace spherical harmonics $Y_{l, m}$.
    """

    name = Quantity(
        type=np.int32,
        description="""
        Angular momentum quantum number of the local orbital.
        """,
    )

    n_wavefunctions = Quantity(
        type=np.int32,
        description="""
        Number of wavefunctions in the l-channel, i.e. $(2l + 1) n_orbitals$.
        """,
    )

    orbitals = SubSection(sub_section=APWWavefunction.m_def, repeats=True)

    def _determine_apw(self, logger: BoundLogger) -> dict[str, int]:
        """
        Produce a count of the APW components in the l-channel.
        """
        count = {'apw': 0, 'lapw': 0, 'slapw': 0, 'lo': 0, 'other': 0}
        order_map = {0: 'apw', 1: 'lapw', 2: 'slapw'}  # TODO: add dirac?

        for orb in self.orbitals:
            err_msg = f'Unknown APW orbital type {orb.type} and order {orb.differential_order}.'
            if isinstance(orb, APWOrbital):
                if orb.type in order_map.values():
                    count[orb.type] += 1
                elif orb.differential_order in order_map:
                    count[order_map[orb.differential_order]] += 1
                elif orb.differential_order > 2:
                    count['slapw'] += 1
                else:
                    logger.warning(err_msg)
            elif isinstance(orb, APWLocalOrbital):
                count['lo'] += 1
            else:
                logger.warning(err_msg)
        return count

    def normalize(self, archive: EntryArchive, logger: BoundLogger) -> None:
        super().normalize(archive, logger)
        # self.name = self.m_def.name
        self.n_wavefunctions = len(self.orbitals) * (2 * self.name + 1)


class MuffinTinRegion(BasisSet, Mesh):
    """
    Muffin-tin region around atoms, containing the augmented part of the APW basis set.
    The latter is structured by l-channel. Each channel contains a base (S)(L)APW definition,
    which may be extended via local orbitals.
    """

    # there are 2 main ways of structuring the APW basis set
    # either as APW and lo in the MT region
    # or by l-channel in the MT region

    l_max = Quantity(
        type=np.int32,
        description="""
        Maximum angular momentum quantum number that is sampled.
        Starts at 0.
        """,
    )

    l_channels = SubSection(sub_section=APWLChannel.m_def, repeats=True)

    def _determine_apw(self, logger: BoundLogger) -> dict[str, int]:
        """
        Aggregate the APW component count in the muffin-tin region.
        """
        count = {'apw': 0, 'lapw': 0, 'slapw': 0, 'lo': 0, 'other': 0}
        for channel in self.l_channels:
            count.update(channel._determine_apw(logger))
        return count

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)
        # self.type = 'spherical'


class BasisSetContainer(NumericalSettings):
    """
    A section defining the full basis set used for representing the electronic structure
    during the diagonalization of a Hamiltonian (component), as defined in `ModelMethod`.
    This section may contain multiple basis set specifications under the basis_set_components,
    each with their own parameters.
    """

    native_tier = Quantity(
        type=str,  # to be overwritten by a parser `MEnum`
        description="""
        Code-specific tag indicating the overall precision based on the basis set parameters.
        The naming conventions of the same code are used. See the parser implementation for the possible values.
        The number of tiers varies, but a typical example would be `low`, `medium`, `high`.
        """,
    )

    # TODO: add reference to `electronic_structure`,
    # specifying to which electronic structure representation the basis set is applied
    # e.g. wavefunction, density, grids for subroutines, etc.

    basis_set_components = SubSection(sub_section=BasisSet.m_def, repeats=True)

    def _determine_apw(self, logger: BoundLogger) -> str:
        """
        Derive the basis set name for a (S)(L)APW case.
        """
        answer, has_plane_wave = '', False
        for comp in self.basis_set_components:
            if isinstance(comp, MuffinTinRegion):
                count = comp._determine_apw(logger)
                if count['apw'] + count['lapw'] + count['slapw'] > 0:
                    if count['slapw'] > 0:
                        answer += 'slapw'.upper()
                    elif count['lapw'] > 0:
                        answer += 'lapw'.upper()
                    elif count['apw'] > 0:
                        answer += 'apw'.upper()
                    if count['lo'] > 0:
                        answer += '+lo'
            elif isinstance(comp, PlaneWaveBasisSet):
                has_plane_wave = True
        return answer if has_plane_wave else ''

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)
        if name := self._determine_apw(logger):
            self.name = name  # TODO: set name based on basis sets
