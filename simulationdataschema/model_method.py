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
from typing import Optional, List
from ase.dft.kpoints import monkhorst_pack

from nomad.units import ureg
from nomad.datamodel.data import ArchiveSection
from nomad.datamodel.metainfo.annotations import ELNAnnotation
from nomad.metainfo import (
    Quantity,
    SubSection,
    MEnum,
    SectionProxy,
    Reference,
    Section,
    Context,
)

from .model_system import ModelSystem
from .atoms_state import OrbitalsState


class NumericalSettings(ArchiveSection):
    """
    A base section used to define the numerical settings used in a simulation. These are meshes,
    self-consistency parameters, and algebraic basis sets.
    """

    name = Quantity(
        type=str,
        description="""
        Name of the numerical settings section. This is typically used to easy identification of the
        `NumericalSettings` section. Possible values: "KMesh", "FrequencyMesh", "TimeMesh",
        "SelfConsistency", "BasisSet".
        """,
        a_eln=ELNAnnotation(component="StringEditQuantity"),
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
        points = None
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

        # Set the name of the section
        self.name = self.m_def.name if self.name is None else self.name

        # If `grid` is not defined, we do not normalize the KMesh
        if self.grid is None:
            logger.warning("Could not find `KMesh.grid`.")
            return

        # Normalize k mesh from grid sampling
        self.points = (
            self.resolve_points(logger) if self.points is None else self.points
        )

        # Calculate k_line_density for data quality measures
        model_systems = self.m_xpath("m_parent.m_parent.model_system", dict=False)
        self.k_line_density = (
            self.resolve_k_line_density(model_systems, logger)
            if self.k_line_density is None
            else self.k_line_density
        )


class FrequencyMesh(Mesh):
    """
    A base section used to specify the settings of a sampling mesh in the frequency real or imaginary space.
    """

    points = Quantity(
        type=np.complex128,
        shape=["n_points", "dimensionality"],
        unit="joule",
        description="""
        List of all the points in the mesh in joules.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        # Set the name of the section
        self.name = self.m_def.name if self.name is None else self.name


class TimeMesh(Mesh):
    """
    A base section used to specify the settings of a sampling mesh in the time real or imaginary space.
    """

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        # Set the name of the section
        self.name = self.m_def.name if self.name is None else self.name


class SelfConsistency(NumericalSettings):
    """
    A base section used to define the parameters used in the self-consistent field (SCF) simulations.
    This section is useful to determine if a simulation `is_converged` or not (see outputs.py) and if
    the simulation fulfills:

        1. The number of iterations is not larger or equal than `n_max_iterations`.

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
        unit="joule",
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
    Model method input parameters and numerical settings used to simulate the materials properties.

    # ! add more description once section is finished
    """

    normalizer_level = 1

    name = Quantity(
        type=str,
        description="""
        Name of the model method. This is typically used to easy identification of the `ModelMethod` section.
        Possible values: "DFT", "TB", "BSE", "GW", "DMFT".
        """,
        a_eln=ELNAnnotation(component="StringEditQuantity"),
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

    numerical_settings = SubSection(sub_section=NumericalSettings.m_def, repeats=True)

    def normalize(self, archive, logger):
        super().normalize(archive, logger)


class ModelMethodElectronic(ModelMethod):
    """
    A base section containing the parameters pertaining to an model Hamiltonian used in electronic structure
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
            "scalar_relativistic",
            "pseudo_scalar_relativistic",
            "scalar_relativistic_atomic_ZORA",
        ),
        description="""
        Describes the relativistic treatment used for the calculation of the final energy
        and related quantities. If `None`, no relativistic treatment is applied.
        """,
    )

    # ? What about this quantity
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

    def normalize(self, archive, logger):
        super().normalize(archive, logger)


class TB(ModelMethodElectronic):
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

    # ? these 2 quantities will change when `BasisSet` is defined
    n_orbitals = Quantity(
        type=np.int32,
        description="""
        Number of orbitals used as a basis to obtain the `TB` model.
        """,
    )

    orbitals_ref = Quantity(
        type=OrbitalsState,
        shape=["n_orbitals"],
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

        We can access the information on the atom by doing:
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
            if self.m_def.name in ["DFTB", "xTB", "Wannier", "SlaterKoster"]
            else None
        )

    def resolve_references_to_states(
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
        # If ModelSystem is not representative, the normalization will not run
        if not model_system.is_representative:
            logger.warning(
                f"`ModelSystem`[{model_index}] section was not found to be representative."
            )
            return None
        # If AtomicCell is not found, the normalization will not run
        atomic_cell = model_system.atomic_cell[0]
        if atomic_cell is None:
            logger.warning("`AtomicCell` section was not found.")
            return None
        # If there is no child ModelSystem, the normalization will not run
        atoms_state = atomic_cell.atoms_state
        model_system_child = model_system.model_system
        if model_system_child is None:
            logger.warning("No child `ModelSystem` section was found.")
            return None
        orbitals_ref = []
        for active_atom in model_system_child:
            # If the child is not an "active_atom", the normalization will not run
            if active_atom.type != "active_atom":
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
        self.name = "TB"

        # Resolve `type` to be defined by the lower level class (Wannier, DFTB, xTB or SlaterKoster) if it is not already defined
        self.type = (
            self.resolve_type(logger)
            if (self.type is None or self.type == "unavailable")
            else self.type
        )

        # Resolve `orbitals_ref` from the info in the child `ModelSystem` section and the `OrbitalsState` sections
        model_systems = self.m_xpath("m_parent.model_system", dict=False)
        if model_systems is None:
            logger.warning(
                "Could not find the `ModelSystem` sections. References to `OrbitalsState` will not be resolved."
            )
            return
        # This normalization only considers the last `ModelSystem` (default `model_index` argument set to -1)
        orbitals_ref = self.resolve_references_to_states(model_systems, logger)
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
        type=MEnum("sss", "sps", "sds"),
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
            "sss": ["s", "s", (0, 0, 0)],
            "sps": ["s", "p", (0, 0, 0)],
            "sds": ["s", "d", (0, 0, 0)],
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
                "The `l_quantum_symbol` of the `OrbitalsState` bonds are not defined."
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
        type=MEnum("dipolar", "quadrupolar", "NRIXS", "Raman"),
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
        the ones happening in quadrupolar, Raman, or NRIXS processes.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        # Add warning in case `multipole_type` and `momentum_transfer` are not consistent
        if (
            self.multipole_type in ["quadrupolar", "NRIXS", "Raman"]
            and self.momentum_transfer is None
        ):
            logger.warning(
                "The `Photon.momentum_transfer` is not defined but the `Photon.multipole_type` describes inelastic scattering processes."
            )
