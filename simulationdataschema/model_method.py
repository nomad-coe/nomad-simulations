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

import numpy as np
import pint
import re
from structlog.stdlib import BoundLogger
from typing import Optional, List, Tuple
from ase.dft.kpoints import monkhorst_pack, get_monkhorst_pack_size_and_offset

from nomad.units import ureg
from nomad.datamodel.data import ArchiveSection
from nomad.datamodel.metainfo.annotations import ELNAnnotation
from nomad.metainfo import (
    Quantity,
    SubSection,
    MEnum,
    Section,
    Context,
    JSON,
)

from .model_system import ModelSystem
from .atoms_state import OrbitalsState, CoreHole
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
        a_eln=ELNAnnotation(component='StringEditQuantity'),
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class Mesh(NumericalSettings):
    """
    A base section used to specify the settings of a sampling mesh. It supports uniformly-spaced
    meshes and symmetry-reduced representations.
    """

    sampling_method = Quantity(
        type=MEnum(
            'Gamma-centered',
            'Monkhorst-Pack',
            'Gamma-offcenter',
            'Line-path',
            'Equidistant',
            'Logarithmic',
            'Tan',
            'Gauss-Legendre',
            'Gauss-Laguerre',
            'Clenshaw-Curtis',
            'Newton-Cotes',
            'Gauss-Hermite',
        ),
        description="""
        Method used to generate the mesh:

        | Name      | Description                      |
        | --------- | -------------------------------- |
        | `'Gamma-centered'` | Regular mesh is centered around Gamma. No offset. |
        | `'Monkhorst-Pack'` | Regular mesh with an offset of half the reciprocal lattice vector. |
        | `'Gamma-offcenter'` | Regular mesh with an offset that is neither `'Gamma-centered'`, nor `'Monkhorst-Pack'`. |
        | `'Line-path'` | Line path along high-symmetry points. Typically employed for simualting band structures. |
        | `'Equidistant'`  | Equidistant 1D grid (also known as 'Newton-Cotes') |
        | `'Logarithmic'`  | log distance 1D grid |
        | `'Tan'`  | Non-uniform tan mesh for 1D grids. More dense at low abs values of the points, while less dense for higher values |
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
        a symmtery operation that was applied to the `Mesh`.
        """,
    )

    # ! is this description correct?
    weights = Quantity(
        type=np.float64,
        shape=['n_points'],
        description="""
        The frequency of times the same point reappears. A value larger than 1, typically
        indicates a symmtery operation that was applied to the mesh.
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
        Offset vector shifting the mesh with respect to a Gamma-centered case.
        """,
    )

    all_points = Quantity(
        type=np.float64,
        shape=['*', 3],
        description="""
        Full list of the mesh points without any symmetry operations.
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

    def resolve_points_and_offset(
        self, logger: BoundLogger
    ) -> Tuple[Optional[List[np.ndarray]], Optional[np.ndarray]]:
        """
        Resolves the `points` and `offset` of the `KMesh` from the `grid` and the `sampling_method`.

        Args:
            logger (BoundLogger): The logger to log messages.

        Returns:
            (Optional[List[pint.Quantity, pint.Quantity]]): The resolved `points` and `offset` of the `KMesh`.
        """
        points = None
        offset = None
        if self.sampling_method == 'Gamma-centered':
            grid_space = [np.linspace(0, 1, n) for n in self.grid]
            points = np.meshgrid(grid_space)
            offset = np.array([0, 0, 0])
        elif self.sampling_method == 'Monkhorst-Pack':
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

            atomic_cell = model_system.atomic_cell
            if atomic_cell is None:
                logger.warning('`ModelSystem.atomic_cell` was not found.')
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

        # Set the name of the section
        self.name = self.m_def.name if self.name is None else self.name

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

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        # Set the name of the section
        self.name = self.m_def.name if self.name is None else self.name


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

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        # Set the name of the section
        self.name = self.m_def.name if self.name is None else self.name


class BasisSet(NumericalSettings):
    """"""

    # TODO work on this base section (@ndaelman-hu)
    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        # Set the name of the section
        self.name = self.m_def.name if self.name is None else self.name


class ModelMethod(ArchiveSection):
    """
    Model method input parameters and numerical settings used to simulate materials properties.

    # ! add more description once section is finished
    """

    normalizer_level = 1

    name = Quantity(
        type=str,
        description="""
        Name of the model method. This is typically used to easy identification of the `ModelMethod` section.

        Suggested values: 'DFT', 'TB', 'GE', 'BSE', 'DMFT', 'NMR', 'kMC'.
        """,
        a_eln=ELNAnnotation(component='StringEditQuantity'),
    )

    type = Quantity(
        type=str,
        description="""
        Identifier used to further specify the type of model Hamiltonian. Example: a TB
        model can be 'Wannier', 'DFTB', 'xTB' or 'Slater-Koster'. This quantity should be
        rewritten to a MEnum when inheriting from this class.
        """,
        a_eln=ELNAnnotation(component='StringEditQuantity'),
    )

    external_reference = Quantity(
        type=str,
        description="""
        External reference to the model e.g. DOI, URL.
        """,
        a_eln=ELNAnnotation(component='URLEditQuantity'),
    )

    numerical_settings = SubSection(sub_section=NumericalSettings.m_def, repeats=True)

    def normalize(self, archive, logger):
        super().normalize(archive, logger)


class ModelMethodElectronic(ModelMethod):
    """
    A base section used to define the parameters of a model Hamiltonian used in electronic structure
    calculations (TB, DFT, GW, BSE, DMFT, etc).
    """

    # ? Is this necessary or will it be defined in another way?
    is_spin_polarized = Quantity(
        type=bool,
        description="""
        If the simulation is done considering the spin degrees of freedom (then there are two spin
        channels, 'down' and 'up') or not.
        """,
    )

    # ? What about this quantity
    relativity_method = Quantity(
        type=MEnum(
            'scalar_relativistic',
            'pseudo_scalar_relativistic',
            'scalar_relativistic_atomic_ZORA',
        ),
        description="""
        Describes the relativistic treatment used for the calculation of the final energy
        and related quantities. If `None`, no relativistic treatment is applied.
        """,
        a_eln=ELNAnnotation(component='EnumEditQuantity'),
    )

    def normalize(self, archive, logger):
        super().normalize(archive, logger)


class XCFunctional(ArchiveSection):
    """
    A base section used to define the parameters of an exchange or correlation functional.
    """

    m_def = Section(validate=False)

    libxc_name = Quantity(
        type=str,
        description="""
        Provides the name of one of the exchange or correlation (XC) functional following the libxc
        convention (see https://www.tddft.org/programs/libxc/).
        """,
        a_eln=ELNAnnotation(component='StringEditQuantity'),
    )

    name = Quantity(
        type=MEnum('exchange', 'correlation', 'hybrid', 'contribution'),
        description="""
        Name of the XC functional. It can be one of the following: 'exchange', 'correlation',
        'hybrid', or 'contribution'.
        """,
    )

    weight = Quantity(
        type=np.float64,
        description="""
        Weight of the functional. This quantity is relevant when defining linear combinations of the
        different functionals. If not specified, its value is 1.
        """,
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )

    # ? add method to extract `name` from `libxc_name`

    def get_weight_name(self, weight: Optional[np.float64]) -> Optional[str]:
        """
        Returns the `weight` as a string with a "*" added at the end.

        Args:
            weight (Optional[np.float64]): The weight of the functional.

        Returns:
            (Optional[str]): The weight as a string with a "*" added at the end.
        """
        weight_name = ''
        if weight is not None:
            weight_name = f'{str(weight)}*'
        return weight_name

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        # Appending `weight` as a string to `libxc_name`
        libxc_name_weight = ''
        if self.weight is not None:
            libxc_name_weight = self.get_weight_name(self.weight)
        if '*' not in self.libxc_name:
            self.libxc_name = libxc_name_weight + self.libxc_name

        # ! check with @ndaelman-hu if this makes sense (COMMENTED OUT FOR NOW)
        # Appending `"+alpha"` in `libxc_name` for hybrids in which the `exact_exchange_mixing_factoris` included
        # libxc_name_alpha = ''
        # if (
        #     self.name == 'hybrid'
        #     and 'exact_exchange_mixing_factor' in self.parameters.keys()
        # ):
        #     libxc_name_alpha = f'+alpha'
        # if '+alpha' not in self.libxc_name:
        #     self.libxc_name = self.libxc_name + libxc_name_alpha


class DFT(ModelMethodElectronic):
    """
    A base section used to define the parameters used in a density functional theory (DFT) calculation.
    """

    # ? Do we need to define `type` for DFT+U?

    jacobs_ladder = Quantity(
        type=MEnum('LDA', 'GGA', 'metaGGA', 'hyperGGA', 'hybrid', 'unavailable'),
        description="""
        Functional classification in line with Jacob's Ladder. See:
            - https://doi.org/10.1063/1.1390175 (original paper)
            - https://doi.org/10.1103/PhysRevLett.91.146401 (meta-GGA)
            - https://doi.org/10.1063/1.1904565 (hyper-GGA)
        """,
    )

    xc_functionals = SubSection(sub_section=XCFunctional.m_def, repeats=True)

    exact_exchange_mixing_factor = Quantity(
        type=np.float64,
        description="""
        Amount of exact exchange mixed in with the XC functional (value range = [0, 1]).
        """,
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )

    # ! MEnum this
    self_interaction_correction_method = Quantity(
        type=str,
        description="""
        Contains the name for the self-interaction correction (SIC) treatment used to
        calculate the final energy and related quantities. If skipped or empty, no special
        correction is applied.

        The following SIC methods are available:

        | SIC method                | Description                       |

        | ------------------------- | --------------------------------  |

        | `""`                      | No correction                     |

        | `"SIC_AD"`                | The average density correction    |

        | `"SIC_SOSEX"`             | Second order screened exchange    |

        | `"SIC_EXPLICIT_ORBITALS"` | (scaled) Perdew-Zunger correction explicitly on a
        set of orbitals |

        | `"SIC_MAURI_SPZ"`         | (scaled) Perdew-Zunger expression on the spin
        density / doublet unpaired orbital |

        | `"SIC_MAURI_US"`          | A (scaled) correction proposed by Mauri and co-
        workers on the spin density / doublet unpaired orbital |
        """,
        a_eln=ELNAnnotation(component='StringEditQuantity'),
    )

    van_der_waals_correction = Quantity(
        type=MEnum('TS', 'OBS', 'G06', 'JCHS', 'MDB', 'XC'),
        description="""
        Describes the Van der Waals (VdW) correction methodology. If `None`, no VdW correction is applied.

        | VdW method  | Reference                               |
        | --------------------- | ----------------------------------------- |
        | `"TS"`  | http://dx.doi.org/10.1103/PhysRevLett.102.073005 |
        | `"OBS"` | http://dx.doi.org/10.1103/PhysRevB.73.205101 |
        | `"G06"` | http://dx.doi.org/10.1002/jcc.20495 |
        | `"JCHS"` | http://dx.doi.org/10.1002/jcc.20570 |
        | `"MDB"` | http://dx.doi.org/10.1103/PhysRevLett.108.236402 and http://dx.doi.org/10.1063/1.4865104 |
        | `"XC"` | The method to calculate the VdW energy uses a non-local functional |
        """,
        a_eln=ELNAnnotation(component='EnumEditQuantity'),
    )

    def __init__(self, m_def: Section = None, m_context: Context = None, **kwargs):
        super().__init__(m_def, m_context, **kwargs)
        self._jacobs_ladder_map = {
            'lda': 'LDA',
            'gga': 'GGA',
            'mgg': 'meta-GGA',
            'hyb_mgg': 'hyper-GGA',
            'hyb': 'hybrid',
        }

    def resolve_libxc_names(
        self, xc_functionals: List[XCFunctional]
    ) -> Optional[List[str]]:
        """
        Resolves the `libxc_names` and sorts them from the list of `XCFunctional` sections.

        Args:
            xc_functionals (List[XCFunctional]): The list of `XCFunctional` sections.

        Returns:
            (Optional[List[str]]): The resolved and sorted `libxc_names`.
        """
        return sorted(
            [
                functional.libxc_name
                for functional in xc_functionals
                if functional.libxc_name is not None
            ]
        )

    def resolve_jacobs_ladder(
        self,
        libxc_names: List[str],
    ) -> str:
        """
        Resolves the `jacobs_ladder` from the `libxc_names`. The mapping (libxc -> NOMAD) is set in `self._jacobs_ladder_map`.

        Args:
            libxc_names (List[str]): The list of `libxc_names`.

        Returns:
            (str): The resolved `jacobs_ladder`.
        """
        if libxc_names is None:
            return 'unavailable'

        rung_order = {x: i for i, x in enumerate(self._jacobs_ladder_map.keys())}
        re_abbrev = re.compile(r'((HYB_)?[A-Z]{3})')

        abbrevs = []
        for xc_name in libxc_names:
            try:
                abbrev = re_abbrev.match(xc_name).group(1)
                abbrev = abbrev.lower() if abbrev == 'HYB_MGG' else abbrev[:3].lower()
                abbrevs.append(abbrev)
            except AttributeError:
                continue

        try:
            highest_rung_abbrev = max(abbrevs, key=lambda x: rung_order[x])
        except KeyError:
            return 'unavailable'
        return self._jacobs_ladder_map.get(highest_rung_abbrev, 'unavailable')

    def resolve_exact_exchange_mixing_factor(
        self, xc_functionals: List[XCFunctional], libxc_names: List[str]
    ) -> Optional[float]:
        """
        Resolves the `exact_exchange_mixing_factor` from the `xc_functionals` and `libxc_names`.

        Args:
            xc_functionals (List[XCFunctional]): The list of `XCFunctional` sections.
            libxc_names (List[str]): The list of `libxc_names`.

        Returns:
            (Optional[float]): The resolved `exact_exchange_mixing_factor`.
        """

        for functional in xc_functionals:
            if functional.name == 'hybrid':
                return functional.parameters.get('exact_exchange_mixing_factor')

        def _scan_patterns(patterns: List[str], xc_name: str) -> bool:
            return any(x for x in patterns if re.search('_' + x + '$', xc_name))

        for xc_name in libxc_names:
            if not re.search('_XC?_', xc_name):
                continue
            if re.search('_B3LYP[35]?$', xc_name):
                return 0.2
            elif _scan_patterns(['HSE', 'PBEH', 'PBE_MOL0', 'PBE_SOL0'], xc_name):
                return 0.25
            elif re.search('_M05$', xc_name):
                return 0.28
            elif re.search('_PBE0_13$', xc_name):
                return 1 / 3
            elif re.search('_PBE38$', xc_name):
                return 3 / 8
            elif re.search('_PBE50$', xc_name):
                return 0.5
            elif re.search('_M06_2X$', xc_name):
                return 0.54
            elif _scan_patterns(['M05_2X', 'PBE_2X'], xc_name):
                return 0.56
        return None

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        libxc_names = self.resolve_libxc_names(self.xc_functionals)
        if libxc_names is not None:
            # Resolves the `jacobs_ladder` from `libxc` mapping
            jacobs_ladder = self.resolve_jacobs_ladder(libxc_names)
            self.jacobs_ladder = (
                jacobs_ladder if self.jacobs_ladder is None else self.jacobs_ladder
            )

            # Resolves the `exact_exchange_mixing_factor` from the `xc_functionals` and `libxc_names`
            if self.xc_functionals is not None:
                exact_exchange_mixing_factor = (
                    self.resolve_exact_exchange_mixing_factor(
                        self.xc_functionals, libxc_names
                    )
                )
                self.exact_exchange_mixing_factor = (
                    exact_exchange_mixing_factor
                    if self.exact_exchange_mixing_factor is None
                    else self.exact_exchange_mixing_factor
                )


class TB(ModelMethodElectronic):
    """
    A base section containing the parameters pertaining to a tight-binding (TB) model calculation.
    The type of tight-binding model is specified in the `type` quantity.
    """

    type = Quantity(
        type=MEnum('DFTB', 'xTB', 'Wannier', 'SlaterKoster', 'unavailable'),
        default='unavailable',
        description="""
        Tight-binding model type.

        | Value | Reference |
        | --------- | ----------------------- |
        | `'DFTB'` | https://en.wikipedia.org/wiki/DFTB |
        | `'xTB'` | https://xtb-docs.readthedocs.io/en/latest/ |
        | `'Wannier'` | https://www.wanniertools.org/theory/tight-binding-model/ |
        | `'SlaterKoster'` | https://journals.aps.org/pr/abstract/10.1103/PhysRev.94.1498 |
        """,
        a_eln=ELNAnnotation(component='EnumEditQuantity'),
    )

    # ? these 2 quantities will change when `BasisSet` is defined
    n_orbitals = Quantity(
        type=np.int32,
        description="""
        Number of orbitals used as a basis to obtain the `TB` model.
        """,
    )

    orbitals_ref = Quantity(
        type=OrbitalsState,
        shape=['n_orbitals'],
        description="""
        References to the `OrbitalsState` sections that contain the orbitals information which are
        relevant for the `TB` model.

        Example: hydrogenated graphene with 3 atoms in the unit cell. The full list of `AtomsState` would
        be
            [
                AtomsState(chemical_symbol='C', orbitals_state=[OrbitalsState('s'), OrbitalsState('px'), OrbitalsState('py'), OrbitalsState('pz')]),
                AtomsState(chemical_symbol='C', orbitals_state=[OrbitalsState('s'), OrbitalsState('px'), OrbitalsState('py'), OrbitalsState('pz')]),
                AtomsState(chemical_symbol='H', orbitals_state=[OrbitalsState('s')]),
            ]

        The relevant orbitals for the TB model are the `'pz'` ones for each `'C'` atom. Then, we define:

            orbitals_ref= [OrbitalState('pz'), OrbitalsState('pz')]

        The relevant atoms information can be accessed from the parent AtomsState sections:
            atom_state = orbitals_ref[i].m_parent
            index = orbitals_ref[i].m_parent_index
            atom_position = orbitals_ref[i].m_parent.m_parent.positions[index]
        """,
    )

    def resolve_type(self, logger: BoundLogger) -> Optional[str]:
        """
        Resolves the `type` of the `TB` section if it is not already defined, and from the
        `m_def.name` of the section.

        Args:
            logger (BoundLogger): The logger to log messages.

        Returns:
            (Optional[str]): The resolved `type` of the `TB` section.
        """
        return (
            self.m_def.name
            if self.m_def.name in ['DFTB', 'xTB', 'Wannier', 'SlaterKoster']
            else None
        )

    def resolve_orbital_references(
        self,
        model_systems: List[ModelSystem],
        logger: BoundLogger,
        model_index: int = -1,
    ) -> Optional[List[OrbitalsState]]:
        """
        Resolves the references to the `OrbitalsState` sections from the child `ModelSystem` section.

        Args:
            model_systems (List[ModelSystem]): The list of `ModelSystem` sections.
            logger (BoundLogger): The logger to log messages.
            model_index (int, optional): The `ModelSystem` section index from which resolve the references. Defaults to -1.

        Returns:
            Optional[List[OrbitalsState]]: The resolved references to the `OrbitalsState` sections.
        """
        model_system = model_systems[model_index]

        # If `ModelSystem` is not representative, the normalization will not run
        if is_not_representative(model_system, logger):
            return None

        # If `AtomicCell` is not found, the normalization will not run
        atomic_cell = model_system.atomic_cell[0]
        if atomic_cell is None:
            logger.warning('`AtomicCell` section was not found.')
            return None

        # If there is no child `ModelSystem`, the normalization will not run
        atoms_state = atomic_cell.atoms_state
        model_system_child = model_system.model_system
        if model_system_child is None:
            logger.warning('No child `ModelSystem` section was found.')
            return None

        # We flatten the `OrbitalsState` sections from the `ModelSystem` section
        orbitals_ref = []
        for active_atom in model_system_child:
            # If the child is not an "active_atom", the normalization will not run
            if active_atom.type != 'active_atom':
                continue
            indices = active_atom.atom_indices
            for index in indices:
                active_atoms_state = atoms_state[index]
                orbitals_state = active_atoms_state.orbitals_state
                for orbital in orbitals_state:
                    orbitals_ref.append(orbital)
        return orbitals_ref

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        # Set `name` to "TB"
        self.name = 'TB'

        # Resolve `type` to be defined by the lower level class (Wannier, DFTB, xTB or SlaterKoster) if it is not already defined
        self.type = (
            self.resolve_type(logger)
            if (self.type is None or self.type == 'unavailable')
            else self.type
        )

        # Resolve `orbitals_ref` from the info in the child `ModelSystem` section and the `OrbitalsState` sections
        model_systems = self.m_xpath('m_parent.model_system', dict=False)
        if model_systems is None:
            logger.warning(
                'Could not find the `ModelSystem` sections. References to `OrbitalsState` will not be resolved.'
            )
            return
        # This normalization only considers the last `ModelSystem` (default `model_index` argument set to -1)
        orbitals_ref = self.resolve_orbital_references(model_systems, logger)
        if orbitals_ref is not None and self.orbitals_ref is None:
            self.n_orbitals = len(orbitals_ref)
            self.orbitals_ref = orbitals_ref


class Wannier(TB):
    """
    A base section used to define the parameters used in a Wannier tight-binding fitting.
    """

    is_maximally_localized = Quantity(
        type=bool,
        description="""
        If the projected orbitals are maximally localized or just a single-shot projection.
        """,
    )

    localization_type = Quantity(
        type=MEnum('single_shot', 'maximally_localized'),
        description="""
        Localization type of the Wannier orbitals.
        """,
    )

    n_bloch_bands = Quantity(
        type=np.int32,
        description="""
        Number of input Bloch bands to calculate the projection matrix.
        """,
    )

    energy_window_outer = Quantity(
        type=np.float64,
        unit='electron_volt',
        shape=[2],
        description="""
        Bottom and top of the outer energy window used for the projection.
        """,
    )

    energy_window_inner = Quantity(
        type=np.float64,
        unit='electron_volt',
        shape=[2],
        description="""
        Bottom and top of the inner energy window used for the projection.
        """,
    )

    def resolve_localization_type(self, logger: BoundLogger) -> Optional[str]:
        """
        Resolves the `localization_type` of the `Wannier` section if it is not already defined, and from the
        `is_maximally_localized` boolean.

        Args:
            logger (BoundLogger): The logger to log messages.

        Returns:
            (Optional[str]): The resolved `localization_type` of the `Wannier` section.
        """
        if self.localization_type is None:
            if self.is_maximally_localized:
                return 'maximally_localized'
            else:
                return 'single_shot'
        logger.info(
            'Could not find if the Wannier tight-binding model is maximally localized or not.'
        )
        return None

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        # Resolve `localization_type` from `is_maximally_localized`
        self.localization_type = self.resolve_localization_type(logger)


class SlaterKosterBond(ArchiveSection):
    """
    A base section used to define the Slater-Koster bond information betwee two orbitals.
    """

    orbital_1 = Quantity(
        type=OrbitalsState,
        description="""
        Reference to the first `OrbitalsState` section.
        """,
    )

    orbital_2 = Quantity(
        type=OrbitalsState,
        description="""
        Reference to the second `OrbitalsState` section.
        """,
    )

    # ? is this the best naming
    bravais_vector = Quantity(
        type=np.int32,
        default=[0, 0, 0],
        shape=[3],
        description="""
        The Bravais vector of the cell in 3 dimensional. This is defined as the vector that connects the
        two atoms that define the Slater-Koster bond. A bond can be defined between orbitals in the
        same unit cell (bravais_vector = [0, 0, 0]) or in neighboring cells (bravais_vector = [m, n, p] with m, n, p are integers).
        Default is [0, 0, 0].
        """,
    )

    # TODO add more names and in the table
    name = Quantity(
        type=MEnum('sss', 'sps', 'sds'),
        description="""
        The name of the Slater-Koster bond. The name is composed by the `l_quantum_symbol` of the orbitals
        and the cell index. Table of possible values:

        | Value   | `orbital_1.l_quantum_symbol` | `orbital_2.l_quantum_symbol` | `bravais_vector` |
        | ------- | ---------------------------- | ---------------------------- | ------------ |
        | `'sss'` | 's' | 's' | [0, 0, 0] |
        | `'sps'` | 's' | 'p' | [0, 0, 0] |
        | `'sds'` | 's' | 'd' | [0, 0, 0] |
        """,
    )

    # ? units
    integral_value = Quantity(
        type=np.float64,
        description="""
        The Slater-Koster bond integral value.
        """,
    )

    def __init__(self, m_def: Section = None, m_context: Context = None, **kwargs):
        super().__init__(m_def, m_context, **kwargs)
        # TODO extend this to cover all bond names
        self._bond_name_map = {
            'sss': ['s', 's', (0, 0, 0)],
            'sps': ['s', 'p', (0, 0, 0)],
            'sds': ['s', 'd', (0, 0, 0)],
        }

    def resolve_bond_name_from_references(
        self,
        orbital_1: OrbitalsState,
        orbital_2: OrbitalsState,
        bravais_vector: tuple,
        logger: BoundLogger,
    ) -> Optional[str]:
        """
        Resolves the `name` of the `SlaterKosterBond` from the references to the `OrbitalsState` sections.

        Args:
            orbital_1 (OrbitalsState): The first `OrbitalsState` section.
            orbital_2 (OrbitalsState): The second `OrbitalsState` section.
            bravais_vector (tuple): The bravais vector of the cell.
            logger (BoundLogger): The logger to log messages.

        Returns:
            (Optional[str]): The resolved `name` of the `SlaterKosterBond`.
        """
        bond_name = None
        if orbital_1.l_quantum_symbol is None or orbital_2.l_quantum_symbol is None:
            logger.warning(
                'The `l_quantum_symbol` of the `OrbitalsState` bonds are not defined.'
            )
            return None
        value = [orbital_1.l_quantum_symbol, orbital_2.l_quantum_symbol, bravais_vector]
        # Check if `value` is found in the `self._bond_name_map` and return the key
        for key, val in self._bond_name_map.items():
            if val == value:
                bond_name = key
                break
        return bond_name

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        if self.orbital_1 and self.orbital_2 and self.bravais_vector is not None:
            bravais_vector = tuple(self.bravais_vector)  # transformed for comparing
            self.name = (
                self.resolve_bond_name_from_references(
                    self.orbital_1, self.orbital_2, bravais_vector, logger
                )
                if self.name is None
                else self.name
            )


class SlaterKoster(TB):
    """
    A base section used to define the parameters used in a Slater-Koster tight-binding fitting.
    """

    bonds = SubSection(sub_section=SlaterKosterBond.m_def, repeats=True)

    overlaps = SubSection(sub_section=SlaterKosterBond.m_def, repeats=True)

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class xTB(TB):
    """
    A base section used to define the parameters used in an extended tight-binding (xTB) calculation.
    """

    # ? Deprecate this

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class Photon(ArchiveSection):
    """
    A base section used to define parameters of a photon, typically used for optical responses.
    """

    # TODO check other options and add specific refs
    multipole_type = Quantity(
        type=MEnum('dipolar', 'quadrupolar', 'NRIXS', 'Raman'),
        description="""
        Type used for the multipolar expansion: dipole, quadrupole, NRIXS, Raman, etc.
        """,
    )

    polarization = Quantity(
        type=np.float64,
        shape=[3],
        description="""
        Direction of the photon polarization in cartesian coordinates.
        """,
    )

    energy = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        Photon energy.
        """,
    )

    momentum_transfer = Quantity(
        type=np.float64,
        shape=[3],
        description="""
        Momentum transfer to the lattice. This quanitity is important for inelastic scatterings, like
        the ones happening in quadrupolar, Raman, or NRIXS processes.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        # Add warning in case `multipole_type` and `momentum_transfer` are not consistent
        if (
            self.multipole_type in ['quadrupolar', 'NRIXS', 'Raman']
            and self.momentum_transfer is None
        ):
            logger.warning(
                'The `Photon.momentum_transfer` is not defined but the `Photon.multipole_type` describes inelastic scattering processes.'
            )


class ExcitedStateMethodology(ModelMethodElectronic):
    """
    A base section used to define the parameters typical of excited-state calculations. "ExcitedStateMethodology"
    mainly refers to methodologies which consider many-body effects as a perturbation of the original
    DFT Hamiltonian. These are: GW, TDDFT, BSE.
    """

    type = Quantity(
        type=str,
        description="""
        Identifier used to further specify the type of model Hamiltonian.
        """,
        a_eln=ELNAnnotation(component='StringEditQuantity'),
    )

    n_states = Quantity(
        type=np.int32,
        description="""
        Number of states used to calculate the excitations.
        """,
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )

    n_empty_states = Quantity(
        type=np.int32,
        description="""
        Number of empty states used to calculate the excitations. This quantity is complementary to `n_states`.
        """,
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )

    broadening = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        Lifetime broadening applied to the spectra in full-width at half maximum for excited-state calculations.
        """,
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )

    q_mesh = SubSection(sub_section=KMesh.m_def)

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class Screening(ExcitedStateMethodology):
    """
    A base section used to define the parameters that define the calculation of screening. This is usually done in
    RPA and linear response.
    """

    dielectric_infinity = Quantity(
        type=np.int32,
        description="""
        Value of the static dielectric constant at infinite q. For metals, this is infinite
        (or a very large value), while for insulators is finite.
        """,
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class GW(ExcitedStateMethodology):
    """
    A base section used to define the parameters of a GW calculation.
    """

    type = Quantity(
        type=MEnum(
            'G0W0',
            'scGW',
            'scGW0',
            'scG0W',
            'ev-scGW0',
            'ev-scGW',
            'qp-scGW0',
            'qp-scGW',
        ),
        description="""
        GW Hedin's self-consistency cycle:

        | Name      | Description                      | Reference             |
        | --------- | -------------------------------- | --------------------- |
        | `'G0W0'`  | single-shot                      | https://journals.aps.org/prb/abstract/10.1103/PhysRevB.74.035101 |
        | `'scGW'`  | self-consistent G and W               | https://journals.aps.org/prb/abstract/10.1103/PhysRevB.75.235102 |
        | `'scGW0'` | self-consistent G with fixed W0  | https://journals.aps.org/prb/abstract/10.1103/PhysRevB.54.8411 |
        | `'scG0W'` | self-consistent W with fixed G0  | -                     |
        | `'ev-scGW0'`  | eigenvalues self-consistent G with fixed W0   | https://journals.aps.org/prb/abstract/10.1103/PhysRevB.34.5390 |
        | `'ev-scGW'`  | eigenvalues self-consistent G and W   | https://journals.aps.org/prb/abstract/10.1103/PhysRevB.74.045102 |
        | `'qp-scGW0'`  | quasiparticle self-consistent G with fixed W0 | https://journals.aps.org/prb/abstract/10.1103/PhysRevB.76.115109 |
        | `'qp-scGW'`  | quasiparticle self-consistent G and W | https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.96.226402 |
        """,
        a_eln=ELNAnnotation(component='EnumEditQuantity'),
    )

    analytical_continuation = Quantity(
        type=MEnum(
            'pade',
            'contour_deformation',
            'ppm_GodbyNeeds',
            'ppm_HybertsenLouie',
            'ppm_vonderLindenHorsh',
            'ppm_FaridEngel',
            'multi_pole',
        ),
        description="""
        Analytical continuation approximations of the GW self-energy:

        | Name           | Description         | Reference                        |
        | -------------- | ------------------- | -------------------------------- |
        | `'pade'` | Pade's approximant  | https://link.springer.com/article/10.1007/BF00655090 |
        | `'contour_deformation'` | Contour deformation | https://journals.aps.org/prb/abstract/10.1103/PhysRevB.67.155208 |
        | `'ppm_GodbyNeeds'` | Godby-Needs plasmon-pole model | https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.62.1169 |
        | `'ppm_HybertsenLouie'` | Hybertsen and Louie plasmon-pole model | https://journals.aps.org/prb/abstract/10.1103/PhysRevB.34.5390 |
        | `'ppm_vonderLindenHorsh'` | von der Linden and P. Horsh plasmon-pole model | https://journals.aps.org/prb/abstract/10.1103/PhysRevB.37.8351 |
        | `'ppm_FaridEngel'` | Farid and Engel plasmon-pole model  | https://journals.aps.org/prb/abstract/10.1103/PhysRevB.47.15931 |
        | `'multi_pole'` | Multi-pole fitting  | https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.74.1827 |
        """,
        a_eln=ELNAnnotation(component='EnumEditQuantity'),
    )

    # TODO improve description
    interval_qp_corrections = Quantity(
        type=np.int32,
        shape=[2],
        description="""
        Band indices (in an interval) for which the GW quasiparticle corrections are calculated.
        """,
    )

    screening_ref = Quantity(
        type=Screening,
        description="""
        Reference to the `Screening` section that the GW calculation used to obtain the screened Coulomb interactions.
        """,
        a_eln=ELNAnnotation(component='ReferenceEditQuantity'),
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class BSE(ExcitedStateMethodology):
    """
    A base section used to define the parameters of a BSE calculation.
    """

    # ? does RPA relates with `screening_ref`?
    type = Quantity(
        type=MEnum('Singlet', 'Triplet', 'IP', 'RPA'),
        description="""
        Type of BSE hamiltonian solved:

            H_BSE = H_diagonal + 2 * gx * Hx - gc * Hc

        where gx, gc specifies the type.

        Online resources for the theory:
        - http://exciting.wikidot.com/carbon-excited-states-from-bse#toc1
        - https://www.vasp.at/wiki/index.php/Bethe-Salpeter-equations_calculations
        - https://docs.abinit.org/theory/bse/
        - https://www.yambo-code.eu/wiki/index.php/Bethe-Salpeter_kernel

        | Name | Description |
        | --------- | ----------------------- |
        | `'Singlet'` | gx = 1, gc = 1 |
        | `'Triplet'` | gx = 0, gc = 1 |
        | `'IP'` | Independent-particle approach |
        | `'RPA'` | Random Phase Approximation |
        """,
        a_eln=ELNAnnotation(component='EnumEditQuantity'),
    )

    solver = Quantity(
        type=MEnum('Full-diagonalization', 'Lanczos-Haydock', 'GMRES', 'SLEPc', 'TDA'),
        description="""
        Solver algotithm used to diagonalize the BSE Hamiltonian.

        | Name | Description | Reference |
        | --------- | ----------------------- | ----------- |
        | `'Full-diagonalization'` | Full diagonalization of the BSE Hamiltonian | - |
        | `'Lanczos-Haydock'` | Subspace iterative Lanczos-Haydock algorithm | https://doi.org/10.1103/PhysRevB.59.5441 |
        | `'GMRES'` | Generalized minimal residual method | https://doi.org/10.1137/0907058 |
        | `'SLEPc'` | Scalable Library for Eigenvalue Problem Computations | https://slepc.upv.es/ |
        | `'TDA'` | Tamm-Dancoff approximation | https://doi.org/10.1016/S0009-2614(99)01149-5 |
        """,
        a_eln=ELNAnnotation(component='EnumEditQuantity'),
    )

    screening_ref = Quantity(
        type=Screening,
        description="""
        Reference to the `Screening` section that the BSE calculation used to obtain the screened Coulomb interactions.
        """,
        a_eln=ELNAnnotation(component='ReferenceEditQuantity'),
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


# ? Is this class really necessary or should go in outputs.py?
class CoreHoleSpectra(ModelMethodElectronic):
    """
    A base section used to define the parameters used in a core-hole spectra calculation. This
    also contains reference to the specific methodological section (DFT, BSE) used to obtain the core-hole spectra.
    """

    m_def = Section(validate=False)

    # # TODO add examples
    # solver = Quantity(
    #     type=str,
    #     description="""
    #     Solver algorithm used for the core-hole spectra.
    #     """,
    #     a_eln=ELNAnnotation(component="StringEditQuantity"),
    # )

    type = Quantity(
        type=MEnum('absorption', 'emission'),
        description="""
        Type of the CoreHole excitation spectra calculated, either "absorption" or "emission".
        """,
        a_eln=ELNAnnotation(component='EnumEditQuantity'),
    )

    edge = Quantity(
        type=MEnum(
            'K',
            'L1',
            'L2',
            'L3',
            'L23',
            'M1',
            'M2',
            'M3',
            'M23',
            'M4',
            'M5',
            'M45',
            'N1',
            'N2',
            'N3',
            'N23',
            'N4',
            'N5',
            'N45',
        ),
        description="""
        Edge label of the excited core-hole. This is obtained by normalization by using `core_hole_ref`.
        """,
    )

    core_hole_ref = Quantity(
        type=CoreHole,
        description="""
        Reference to the `CoreHole` section that contains the information of the edge of the excited core-hole.
        """,
        a_eln=ELNAnnotation(component='ReferenceEditQuantity'),
    )

    excited_state_method_ref = Quantity(
        type=ModelMethodElectronic,
        description="""
        Reference to the `ModelMethodElectronic` section (e.g., `DFT` or `BSE`) that was used to obtain the core-hole spectra.
        """,
        a_eln=ELNAnnotation(component='ReferenceEditQuantity'),
    )

    # TODO add normalization to obtain `edge`

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class DMFT(ModelMethodElectronic):
    """
    A base section used to define the parameters of a DMFT calculation.
    """

    impurity_solver = Quantity(
        type=MEnum(
            'CT-INT',
            'CT-HYB',
            'CT-AUX',
            'ED',
            'NRG',
            'MPS',
            'IPT',
            'NCA',
            'OCA',
            'slave_bosons',
            'hubbard_I',
        ),
        description="""
        Impurity solver method used in the DMFT loop:

        | Name              | Reference                            |
        | ----------------- | ------------------------------------ |
        | `'CT-INT'`        | https://link.springer.com/article/10.1134/1.1800216 |
        | `'CT-HYB'`        | https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.97.076405 |
        | `'CT-AUX'`        | https://iopscience.iop.org/article/10.1209/0295-5075/82/57003 |
        | `'ED'`            | https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.72.1545 |
        | `'NRG'`           | https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.80.395 |
        | `'MPS'`           | https://journals.aps.org/prb/abstract/10.1103/PhysRevB.90.045144 |
        | `'IPT'`           | https://journals.aps.org/prb/abstract/10.1103/PhysRevB.45.6479 |
        | `'NCA'`           | https://journals.aps.org/prb/abstract/10.1103/PhysRevB.47.3553 |
        | `'OCA'`           | https://journals.aps.org/prb/abstract/10.1103/PhysRevB.47.3553 |
        | `'slave_bosons'`  | https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.57.1362 |
        | `'hubbard_I'`     | https://iopscience.iop.org/article/10.1088/0953-8984/24/7/075604 |
        """,
    )

    n_impurities = Quantity(
        type=np.int32,
        description="""
        Number of impurities mapped from the correlated atoms in the unit cell. This defines whether
        the DMFT calculation is done in a single-impurity or multi-impurity run.
        """,
    )

    n_orbitals = Quantity(
        type=np.int32,
        shape=['n_impurities'],
        description="""
        Number of correlated orbitals per impurity.
        """,
    )

    orbitals_ref = Quantity(
        type=OrbitalsState,
        shape=['n_orbitals'],
        description="""
        References to the `OrbitalsState` sections that contain the orbitals information which are
        relevant for the `DMFT` calculation.

        Example: hydrogenated graphene with 3 atoms in the unit cell. The full list of `AtomsState` would
        be
            [
                AtomsState(chemical_symbol='C', orbitals_state=[OrbitalsState('s'), OrbitalsState('px'), OrbitalsState('py'), OrbitalsState('pz')]),
                AtomsState(chemical_symbol='C', orbitals_state=[OrbitalsState('s'), OrbitalsState('px'), OrbitalsState('py'), OrbitalsState('pz')]),
                AtomsState(chemical_symbol='H', orbitals_state=[OrbitalsState('s')]),
            ]

        The relevant orbitals for the TB model are the `'pz'` ones for each `'C'` atom. Then, we define:

            orbitals_ref= [OrbitalState('pz'), OrbitalsState('pz')]

        The relevant impurities information can be accesed from the parent AtomsState sections:
            impurity_state = orbitals_ref[i].m_parent
            index = orbitals_ref[i].m_parent_index
            impurity_position = orbitals_ref[i].m_parent.m_parent.positions[index]
        """,
    )

    # ? Improve this with `orbitals_ref.occupation` and possibly a function?
    n_electrons = Quantity(
        type=np.float64,
        shape=['n_impurities'],
        description="""
        Initial number of valence electrons per impurity.
        """,
    )

    inverse_temperature = Quantity(
        type=np.float64,
        unit='1/joule',
        description="""
        Inverse temperature = 1/(kB*T).
        """,
    )

    # ? Check this once magnetic states are better covered in the schema. This will be probably under `ModelSystem`
    # ? by checking the spins in `AtomsState` for the `AtomicCell`
    # ! Check solid_dmft example by using magmom (atomic magnetic moments), and improve on AtomsState to include such moments
    magnetic_state = Quantity(
        type=MEnum('paramagnetic', 'ferromagnetic', 'antiferromagnetic'),
        description="""
        Magnetic state in which the DMFT calculation is done. This quantity can be obtained from
        `orbitals_ref` and their spin state.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)
