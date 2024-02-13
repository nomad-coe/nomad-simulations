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
from ase.dft.kpoints import monkhorst_pack

from nomad.units import ureg
from nomad.datamodel.data import ArchiveSection
from nomad.datamodel.metainfo.annotations import ELNAnnotation
from nomad.metainfo import Quantity, SubSection, MEnum, SectionProxy, Reference


class Mesh(ArchiveSection):
    """
    A base section used to specify the settings of a sampling mesh. It supports uniformly-spaced
    meshes and symmetry-reduced representations.
    """

    dimensionality = Quantity(
        type=np.int32,
        description="""
        Dimensionality of the mesh: 1, 2, or 3.
        """,
    )

    sampling_method = Quantity(
        type=MEnum(
            "Gamma-centered",
            "Monkhorst-Pack",
            "Gamma-offcenter",
            "Line-path",
            "Equidistant",
            "Logarithmic",
            "Tan",
            "Gauss-Legendre",
            "Gauss-Laguerre" "Clenshaw-Curtis",
            "Newton-Cotes",
            "Gauss-Hermite",
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
        Total number of points in the mesh, accounting for the multiplicities.
        """,
    )

    grid = Quantity(
        type=np.int32,
        shape=["dimensionality"],
        description="""
        Amount of mesh point sampling along each axis, i.e. [nx, ny, nz].
        """,
    )

    points = Quantity(
        type=np.complex128,
        shape=["*", "dimensionality"],
        description="""
        List of all the points in the mesh.
        """,
    )

    multiplicities = Quantity(
        type=np.float64,
        shape=["*"],
        description="""
        The amount of times the same point reappears. These are accounted for in `n_points`.
        A value larger than 1, typically indicates a symmtery operation that was applied to the mesh.
        """,
    )

    weights = Quantity(
        type=np.float64,
        shape=["*"],
        description="""
        The frequency of times the same point reappears.
        A value larger than 1, typically indicates a symmtery operation that was applied to the mesh.
        """,
    )

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        self.dimensionality = 3 if not self.dimensionality else self.dimensionality
        if self.grid is None:
            return
        self.n_points = np.prod(self.grid) if not self.n_points else self.n_points


class LinePathSegment(ArchiveSection):
    """
    A base section used to define the settings of a single line path segment within a
    multidimensional mesh.
    """

    start_point = Quantity(
        type=str,
        description="""
        Name of the hihg-symmetry starting point of the line path segment.
        """,
    )

    end_point = Quantity(
        type=str,
        description="""
        Name of the high-symmetry end point of the line path segment.
        """,
    )

    n_points = Quantity(
        type=np.int32,
        description="""
        Number of points in the line path segment.
        """,
    )

    points = Quantity(
        type=np.float64,
        shape=["*", 3],
        description="""
        List of all the points in the line path segment.
        """,
    )


class KMesh(Mesh):
    """
    A base section used to specify the settings of a sampling mesh in reciprocal space.
    """

    reciprocal_lattice_vectors = Quantity(
        type=np.float64,
        shape=[3, 3],
        unit="1/meter",
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
        shape=["*", 3],
        description="""
        Full list of the mesh points without any symmetry operations.
        """,
    )

    high_symmetry_points = Quantity(
        type=str,
        shape=["*"],
        description="""
        Named high symmetry points in the mesh.
        """,
    )

    k_line_density = Quantity(
        type=np.float64,
        unit="m",
        description="""
        Amount of sampled k-points per unit reciprocal length along each axis.
        Contains the least precise density out of all axes.
        Should only be compared between calulations of similar dimensionality.
        """,
    )

    line_path_segments = SubSection(sub_section=LinePathSegment.m_def, repeats=True)

    def get_k_line_density(self, reciprocal_lattice_vectors):
        """
        Calculates the k-line density of the KMesh. This quantity is used to have an idea
        of the precision of the KMesh sampling.

        Args:
            reciprocal_lattice_vectors (np.array): Reciprocal lattice vectors of the
            atomic cell.

        Returns:
            (np.float64): The k-line density of the KMesh.
        """
        if reciprocal_lattice_vectors is None:
            return
        if len(reciprocal_lattice_vectors) != 3 or len(self.grid) != 3:
            return None

        reciprocal_lattice_vectors = reciprocal_lattice_vectors.magnitude
        return min(
            [
                k_point / (np.linalg.norm(k_vector))
                for k_vector, k_point in zip(reciprocal_lattice_vectors, self.grid)
            ]
        )

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        # If `grid` is not defined, we do not normalize the KMesh
        if self.grid is None:
            return

        # Normalize k mesh from grid sampling
        if self.sampling_method == "Gamma-centered":
            self.points = np.meshgrid(*[np.linspace(0, 1, n) for n in self.grid])
        elif self.sampling_method == "Monkhorst-Pack":
            try:
                self.points += monkhorst_pack(self.grid)
            except ValueError:
                pass  # this is a quick workaround: k_mesh.grid should be symmetry reduced

        # Calculate k_line_density for precision
        model_system = self.m_xpath("m_parent.m_parent.model_system[-1]", dict=False)
        if self.k_line_density is None and model_system is not None:
            if not model_system.is_representative:
                logger.warning(
                    "The last ModelSystem was not found to be representative. We will not "
                    "extract k_line_density."
                )
                return
            if model_system.type != "bulk":
                logger.warning(
                    "ModelSystem is not a bulk system. We will not extract k_line_density."
                )
                return
            atomic_cell = model_system.atomic_cell
            if atomic_cell is None:
                logger.warning(
                    "Atomic cell was not found in the ModelSystem. We will not extract "
                    "k_line_density."
                )
                return
            ase_atoms = atomic_cell[0].to_ase_atoms(logger)
            if self.reciprocal_lattice_vectors is None:
                self.reciprocal_lattice_vectors = (
                    2 * np.pi * ase_atoms.get_reciprocal_cell() / ureg.angstrom
                )
            if k_line_density := self.get_k_line_density(
                self.reciprocal_lattice_vectors
            ):
                self.k_line_density = k_line_density * ureg("m")


class FrequencyMesh(Mesh):
    """
    A base section used to specify the settings of a sampling mesh in the 1D frequency
    real or imaginary space.
    """

    points = Quantity(
        type=np.complex128,
        shape=["*", "dimensionality"],
        unit="joule",
        description="""
        List of all the points in the mesh in joules.
        """,
    )

    smearing = Quantity(
        type=np.float64,
        description="""
        Numerical smearing parameter used for convolutions.
        """,
    )

    def normalize(self, archive, logger):
        super().normalize(archive, logger)


class TimeMesh(Mesh):
    """
    A base section used to specify the settings of a sampling mesh in the 1D time real or
    imaginary space.
    """

    smearing = Quantity(
        type=np.float64,
        description="""
        Numerical smearing parameter used for convolutions.
        """,
    )

    def normalize(self, archive, logger):
        super().normalize(archive, logger)


class ModelMethod(ArchiveSection):
    """ """

    normalizer_level = 1

    name = Quantity(
        type=MEnum("DFT", "TB", "GW", "DMFT", "BSE", "kMC", "unavailable"),
        default="unavailable",
        description="""
        Name of the model used to simulate the materials properties.

        | Value | Reference |
        | --------- | ----------------------- |
        | `'DFT'` | https://en.wikipedia.org/wiki/Density_functional_theory |
        | `'TB'` | https://en.wikipedia.org/wiki/Tight_binding |
        | `'GW'` | https://en.wikipedia.org/wiki/GW_approximation |
        | `'DMFT'` | https://en.wikipedia.org/wiki/Dynamical_mean-field_theory |
        | `'BSE'` | https://en.wikipedia.org/wiki/Bethe-Salpeter_equation |
        | `'kMC'` | https://en.wikipedia.org/wiki/Kinetic_Monte_Carlo |
        """,
        a_eln=ELNAnnotation(component="EnumEditQuantity"),
    )

    type = Quantity(
        type=str,
        description="""
        Identifyer used to further specify the type of model Hamiltonian. Example: a TB
        model can be 'Wannier', 'DFTB', 'xTB' or 'Slater-Koster'. This quantity should be
        rewritten to a MEnum when inheriting from this class.
        """,
        a_eln=ELNAnnotation(component="StringEditQuantity"),
    )

    reference = Quantity(
        type=str,
        shape=[],
        description="""
        Reference to the model e.g. DOI, URL.
        """,
        a_eln=ELNAnnotation(component="URLEditQuantity"),
    )

    starting_method_ref = Quantity(
        type=Reference(SectionProxy("ModelMethod")),
        description="""
        Links the current section method to a section method containing the starting
        parameters.
        """,
    )

    k_mesh = SubSection(sub_section=KMesh.m_def)

    frequency_mesh = SubSection(sub_section=FrequencyMesh.m_def, repeats=True)

    time_mesh = SubSection(sub_section=TimeMesh.m_def, repeats=True)

    relativity_method = Quantity(
        type=MEnum(
            "scalar_relativistic",
            "pseudo_scalar_relativistic",
            "scalar_relativistic_atomic_ZORA",
        ),
        description="""
        Describes the relativistic treatment used for the calculation of the final energy
        and related quantities. If empty, no relativistic treatment is applied.
        """,
    )

    van_der_waals_method = Quantity(
        type=MEnum("TS", "OBS", "G06", "JCHS", "MDB", "XC"),
        description="""
        Describes the Van der Waals method. If empty, no Van der Waals correction is applied.

        | Van der Waals method  | Reference                               |

        | --------------------- | ----------------------------------------- |

        | `"TS"`  | http://dx.doi.org/10.1103/PhysRevLett.102.073005 |

        | `"OBS"` | http://dx.doi.org/10.1103/PhysRevB.73.205101 |

        | `"G06"` | http://dx.doi.org/10.1002/jcc.20495 |

        | `"JCHS"` | http://dx.doi.org/10.1002/jcc.20570 |

        | `"MDB"` | http://dx.doi.org/10.1103/PhysRevLett.108.236402 and http://dx.doi.org/10.1063/1.4865104 |

        | `"XC"` | The method to calculate the Van der Waals energy uses a non-local functional |
        """,
    )

    def resolve_model_method_name(self, logger):
        """
        Resolves the name of the ModelMethod section if it is not already defined from the
        section inheriting from this one.
        """
        if self.m_def.inherited_sections is None:
            logger.warning(
                "Could not find if the ModelMethod section used is inheriting from another section."
            )
            return
        inherited_section = self.m_def.inherited_sections[2]
        if inherited_section.name == "TB":
            self.name = inherited_section.name

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        # We go 2 levels down with respect to ArchiveSection to extract `name` if it is not already defined
        if self.name is None or self.name == "unavailable":
            self.resolve_model_method_name(logger)

        # If `starting_method_ref` does not exist, we define it with respect to the initial section
        if self.starting_method_ref is None and self.m_parent_index > 0:
            self.starting_method_ref = self.m_parent.model_method[0]


class TB(ModelMethod):
    """
    A base section containing the parameters pertaining to a tight-binding model calculation.
    The type of tight-binding model is specified in the `type` quantity.
    """

    type = Quantity(
        type=MEnum("DFTB", "xTB", "Wannier", "SlaterKoster", "unavailable"),
        default="unavailable",
        description="""
        Tight-binding model type.

        | Value | Reference |
        | --------- | ----------------------- |
        | `'DFTB'` | https://en.wikipedia.org/wiki/DFTB |
        | `'xTB'` | https://xtb-docs.readthedocs.io/en/latest/ |
        | `'Wannier'` | https://www.wanniertools.org/theory/tight-binding-model/ |
        | `'SlaterKoster'` | https://journals.aps.org/pr/abstract/10.1103/PhysRev.94.1498 |
        """,
        a_eln=ELNAnnotation(component="EnumEditQuantity"),
    )

    n_orbitals = Quantity(
        type=np.int32,
        description="""
        Number of orbitals used as a basis to obtain the tight-binding model Hamiltonian.
        """,
    )

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        # We set the `type` to be defined by the lower level class (Wannier, DFTB, xTB or SlaterKoster)
        # if it is not already defined
        if self.type is None or self.type == "unavailable":
            self.type = self.m_def.name


class Wannier(TB):
    """
    A base section containing the parameters pertaining to a Wannier tight-binding model
    calculation.
    """

    is_maximally_localized = Quantity(
        type=bool,
        description="""
        If the projected orbitals are maximally localized or just a single-shot projection.
        """,
    )

    n_bloch_bands = Quantity(
        type=np.int32,
        description="""
        Number of input Bloch bands to calculate the projection matrix.
        """,
    )

    convergence_tolerance_max_localization = Quantity(
        type=np.float64,
        description="""
        Convergence tolerance for maximal localization of the projected orbitals.
        """,
    )

    energy_window_outer = Quantity(
        type=np.float64,
        unit="electron_volt",
        shape=[2],
        description="""
        Bottom and top of the outer energy window used for the projection.
        """,
    )

    energy_window_inner = Quantity(
        type=np.float64,
        unit="electron_volt",
        shape=[2],
        description="""
        Bottom and top of the inner energy window used for the projection.
        """,
    )

    def normalize(self, archive, logger):
        super().normalize(archive, logger)
