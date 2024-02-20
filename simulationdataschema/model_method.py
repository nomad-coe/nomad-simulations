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
from structlog.stdlib import BoundLogger
from typing import Optional, List, Tuple
from ase.dft.kpoints import monkhorst_pack

from nomad.units import ureg
from nomad.datamodel.data import ArchiveSection
from nomad.datamodel.metainfo.annotations import ELNAnnotation
from nomad.metainfo import Quantity, SubSection, MEnum, SectionProxy, Reference

from .model_system import ModelSystem


class Mesh(ArchiveSection):
    """
    A base section used to specify the settings of a sampling mesh. It supports uniformly-spaced
    meshes and symmetry-reduced representations.
    """

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
        shape=["dimensionality"],
        description="""
        Amount of mesh point sampling along each axis, i.e. [nx, ny, nz].
        """,
    )

    points = Quantity(
        type=np.complex128,
        shape=["n_points", "dimensionality"],
        description="""
        List of all the points in the mesh.
        """,
    )

    multiplicities = Quantity(
        type=np.float64,
        shape=["n_points"],
        description="""
        The amount of times the same point reappears. A value larger than 1, typically indicates
        a symmtery operation that was applied to the `Mesh`.
        """,
    )

    # ! is this description correct?
    weights = Quantity(
        type=np.float64,
        shape=["n_points"],
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

    start_point = Quantity(
        type=str,
        description="""
        Name of the high-symmetry starting point of the line path segment.
        """,
    )

    end_point = Quantity(
        type=str,
        description="""
        Name of the high-symmetry end point of the line path segment.
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
        shape=["n_line_points", 3],
        description="""
        List of all the points in the line path segment.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


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

    def resolve_points(self, logger: BoundLogger) -> Optional[pint.Quantity]:
        """
        Resolves the `points` of the `KMesh` from the `grid` and the `sampling_method`.

        Args:
            logger (BoundLogger): The logger to log messages.

        Returns:
            (Optional[pint.Quantity]): The resolved `points` of the `KMesh`.
        """
        if self.sampling_method == "Gamma-centered":
            points = np.meshgrid(*[np.linspace(0, 1, n) for n in self.grid])
        elif self.sampling_method == "Monkhorst-Pack":
            try:
                points += monkhorst_pack(self.grid)
            except ValueError:
                logger.warning(
                    "Could not resolve `KMesh.points` from `KMesh.grid`. Monkhorst-Pack failed."
                )
                return None  # this is a quick workaround: k_mesh.grid should be symmetry reduced
        return points

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
            logger.error("No `reciprocal_lattice_vectors` input found.")
            return None
        if len(reciprocal_lattice_vectors) != 3 or len(self.grid) != 3:
            logger.error(
                "The `reciprocal_lattice_vectors` and the `grid` should have the same dimensionality."
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
            if not model_system.is_representative:
                logger.warning(
                    "The last `ModelSystem` was not found to be representative."
                )
                return None
            if model_system.type != "bulk":
                logger.warning("`ModelSystem.type` is not describing a bulk system.")
                return None
            atomic_cell = model_system.atomic_cell
            if atomic_cell is None:
                logger.warning("`ModelSystem.atomic_cell` was not found.")
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
                return k_line_density * ureg("m")
        return None

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        # If `grid` is not defined, we do not normalize the KMesh
        if self.grid is None:
            return

        # Normalize k mesh from grid sampling
        points = self.resolve_points(logger)
        if self.points is None and points is not None:
            self.points = points

        # Calculate k_line_density for data quality measures
        model_systems = self.m_xpath("m_parent.m_parent.model_system", dict=False)
        k_line_density = self.resolve_k_line_density(model_systems, logger)
        if self.k_line_density is None and k_line_density is not None:
            self.k_line_density = k_line_density


class FrequencyMesh(Mesh):
    """
    A base section used to specify the settings of a sampling mesh in the 1D frequency real or imaginary space.
    """

    points = Quantity(
        type=np.complex128,
        shape=["n_points", "dimensionality"],
        unit="joule",
        description="""
        List of all the points in the mesh in joules.
        """,
    )

    smearing_width = Quantity(
        type=np.float64,
        description="""
        Numerical smearing parameter used for convolutions.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class TimeMesh(Mesh):
    """
    A base section used to specify the settings of a sampling mesh in the 1D time real or imaginary space.
    """

    smearing_width = Quantity(
        type=np.float64,
        description="""
        Numerical smearing parameter used for convolutions.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class SelfConsistency(ArchiveSection):
    """
    A base section used to define the parameters used in the self-consistent field (SCF) simulations.
    This section is useful to determine if a simulation `is_converged` or not (see outputs.py) and if
    the simulation fulfills:

        1. The number of iteractions is not larger or equal than `n_max_iterations`.

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

    threshold_energy_change = Quantity(
        type=np.float64,
        unit="joule",
        description="""
        Specifies the threshold for the total energy change between two subsequent self-consistent iterations.
        The simulation `is_converged` if the total energy change between two SCF cycles is below
        this threshold.
        """,
    )

    # ? add unit
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


class ModelMethod(ArchiveSection):
    """
    Model method input parameters and settings used to simulate the materials properties.

    # ! add more description once section is finished
    """

    normalizer_level = 1

    name = Quantity(
        type=MEnum("DFT", "TB", "GW", "DMFT", "BSE", "kMC", "NMR", "unavailable"),
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
        | `'NMR'` | https://en.wikipedia.org/wiki/Nuclear_magnetic_resonance |
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

    external_reference = Quantity(
        type=str,
        description="""
        External reference to the model e.g. DOI, URL.
        """,
        a_eln=ELNAnnotation(component="URLEditQuantity"),
    )

    n_method_references = Quantity(
        type=np.int32,
        description="""
        Number of other `ModelMethod` section references of the current method.
        """,
    )

    methods_ref = Quantity(
        type=Reference(SectionProxy("ModelMethod")),
        shape=["n_method_references"],
        description="""
        Links the current `ModelMethod` section to other `ModelMethod` sections. For example, a
        `BSE` calculation is composed of two `ModelMethod` sections: one with the settings of the
        incoming `Photon`, and another with the settings of the `BSE` calculation.
        """,
    )

    k_mesh = SubSection(sub_section=KMesh.m_def)

    frequency_mesh = SubSection(sub_section=FrequencyMesh.m_def, repeats=True)

    time_mesh = SubSection(sub_section=TimeMesh.m_def, repeats=True)

    self_consistency = SubSection(sub_section=SelfConsistency.m_def)

    is_spin_polarized = Quantity(
        type=bool,
        description="""
        If the simulation is done considering the spin degrees of freedom (then there are two spin
        channels, 'down' and 'up') or not.
        """,
    )

    # ? What about this quantity?
    relativity_method = Quantity(
        type=MEnum(
            "scalar_relativistic",
            "pseudo_scalar_relativistic",
            "scalar_relativistic_atomic_ZORA",
        ),
        description="""
        Describes the relativistic treatment used for the calculation of the final energy
        and related quantities. If `None`, no relativistic treatment is applied.
        """,
    )

    # ? What about this quantity?
    van_der_waals_method = Quantity(
        type=MEnum("TS", "OBS", "G06", "JCHS", "MDB", "XC"),
        description="""
        Describes the Van der Waals method. If `None`, no Van der Waals correction is applied.

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

    def resolve_methods_ref(
        self, logger: BoundLogger
    ) -> Tuple[np.int32, List[ArchiveSection]]:
        """
        Resolves the list of other `ModelMethod` sections that the current `ModelMethod` section will be referencing
        to from the parent section.

        Args:
            logger (BoundLogger): The logger to log messages.

        Returns:
            (List[ArchiveSection]): The list of referenced `ModelMethod` sections.
        """
        n_method_references = self.m_parent_index - 1
        methods_ref = []
        for index in range(n_method_references):
            methods_ref.append(self.m_parent.model_method[index])
        return n_method_references, methods_ref

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        # If `methods_ref` does not exist but there are multiple `ModelMethod` sections in the parent, we define refs to them.
        if self.methods_ref is None and self.m_parent_index > 0:
            self.n_method_references, self.methods_ref = self.resolve_methods_ref(
                logger
            )


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
            if self.m_def.name in ["DFTB", "xTB", "Wannier", "SlaterKoster"]
            else self.type
        )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        # Set `name` to "TB"
        self.name = "TB"

        # Resolve `type` to be defined by the lower level class (Wannier, DFTB, xTB or SlaterKoster) if it is not already defined
        self.type = self.resolve_type(logger)


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
        type=MEnum("single_shot", "maximally_localized"),
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
                return "maximally_localized"
            else:
                return "single_shot"
        logger.info(
            "Could not find if the Wannier tight-binding model is maximally localized or not."
        )
        return None

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        # Resolve `localization_type` from `is_maximally_localized`
        self.localization_type = self.resolve_localization_type(logger)


class TightBindingOrbital(ArchiveSection):
    """
    A base section used to define an orbital paramters, including the name of orbital, shell number and the on-site energy.
    """

    orbital_name = Quantity(
        type=str,
        description="""
        The name of the orbital.
        """,
    )

    cell_index = Quantity(
        type=np.int32,
        shape=[3],
        description="""
        The index of the cell in 3 dimensional.
        """,
    )

    atom_index = Quantity(
        type=np.int32,
        description="""
        The index of the atom.
        """,
    )

    shell = Quantity(
        type=np.int32,
        description="""
        The shell number.
        """,
    )

    onsite_energy = Quantity(
        type=np.float64,
        description="""
        On-site energy of the orbital.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class TwoCenterBond(ArchiveSection):
    """
    A base section used to define the two-center-approximation bonds between two atoms.
    """

    # TODO add examples
    bond_label = Quantity(
        type=str,
        description="""
        Label to identify the Slater-Koster bond.
        """,
    )

    # ! Improve naming
    center1 = SubSection(
        sub_section=TightBindingOrbital.m_def,
        repeats=False,
        description="""
        Name of the Slater-Koster bond to identify the bond.
        """,
    )

    # ! Improve naming
    center2 = SubSection(
        sub_section=TightBindingOrbital.m_def,
        repeats=False,
        description="""
        Name of the Slater-Koster bond to identify the bond.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class SlaterKosterBond(TwoCenterBond):
    """
    A base section used to define the Slater-Koster integrals for two orbitals between two atoms.
    """

    sss = Quantity(
        type=np.float64,
        description="""
        The Slater-Koster integral of type sigma between two s orbitals.
        """,
    )

    sps = Quantity(
        type=np.float64,
        description="""
        The Slater-Koster integral of type sigma between s and p orbitals.
        """,
    )

    sds = Quantity(
        type=np.float64,
        description="""
        The Slater-Koster integral of type sigma between s and d orbitals.
        """,
    )

    sfs = Quantity(
        type=np.float64,
        description="""
        The Slater-Koster integral of type sigma between s and f orbitals.
        """,
    )

    pps = Quantity(
        type=np.float64,
        description="""
        The Slater-Koster integral of type sigma between two p orbitals.
        """,
    )

    ppp = Quantity(
        type=np.float64,
        description="""
        The Slater-Koster integral of type pi between two p orbitals.
        """,
    )

    pds = Quantity(
        type=np.float64,
        description="""
        The Slater-Koster integral of type sigma between p and d orbitals.
        """,
    )

    pdp = Quantity(
        type=np.float64,
        description="""
        The Slater-Koster integral of type pi between p and d orbitals.
        """,
    )

    pfs = Quantity(
        type=np.float64,
        description="""
        The Slater-Koster integral of type sigma between p and f orbitals.
        """,
    )

    pfp = Quantity(
        type=np.float64,
        description="""
        The Slater-Koster integral of type pi between p and f orbitals.
        """,
    )

    dds = Quantity(
        type=np.float64,
        description="""
        The Slater-Koster integral of type sigma between two d orbitals.
        """,
    )

    ddp = Quantity(
        type=np.float64,
        description="""
        The Slater-Koster integral of type pi between two d orbitals.
        """,
    )

    ddd = Quantity(
        type=np.float64,
        description="""
        The Slater-Koster integral of type delta between two d orbitals.
        """,
    )

    dfs = Quantity(
        type=np.float64,
        description="""
        The Slater-Koster integral of type sigma between d and f orbitals.
        """,
    )

    dfp = Quantity(
        type=np.float64,
        description="""
        The Slater-Koster integral of type pi between d and f orbitals.
        """,
    )

    dfd = Quantity(
        type=np.float64,
        description="""
        The Slater-Koster integral of type delta between d and f orbitals.
        """,
    )

    ffs = Quantity(
        type=np.float64,
        description="""
        The Slater-Koster integral of type sigma between two f orbitals.
        """,
    )

    ffp = Quantity(
        type=np.float64,
        description="""
        The Slater-Koster integral of type pi between two f orbitals.
        """,
    )

    ffd = Quantity(
        type=np.float64,
        description="""
        The Slater-Koster integral of type delta between two f orbitals.
        """,
    )

    fff = Quantity(
        type=np.float64,
        description="""
        The Slater-Koster integral of type phi between two f orbitals.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class SlaterKoster(TB):
    """
    A base section used to define the parameters used in a Slater-Koster tight-binding fitting.
    """

    orbitals = SubSection(sub_section=TightBindingOrbital.m_def, repeats=True)

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

    multipole_type = Quantity(
        type=str,
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
        unit="joule",
        description="""
        Photon energy.
        """,
    )

    momentum_transfer = Quantity(
        type=np.float64,
        shape=[3],
        description="""
        Momentum transfer to the lattice. This quanitity is important for inelastic scatterings, like
        the ones happening in Raman or NRIXS.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)
