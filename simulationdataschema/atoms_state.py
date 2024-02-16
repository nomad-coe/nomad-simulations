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
import ase
from typing import Optional
from structlog.stdlib import BoundLogger

from nomad.units import ureg

from nomad.metainfo import Quantity, SubSection, MEnum
from nomad.datamodel.data import ArchiveSection
from nomad.datamodel.metainfo.annotations import ELNAnnotation


orbitals = {
    -1: dict(zip(range(4), ("s", "p", "d", "f"))),
    0: {0: ""},
    1: dict(zip(range(-1, 2), ("x", "z", "y"))),
    2: dict(zip(range(-2, 3), ("xy", "xz", "z^2", "yz", "x^2-y^2"))),
    3: dict(
        zip(
            range(-3, 4),
            ("x(x^2-3y^2)", "xyz", "xz^2", "z^3", "yz^2", "z(x^2-y^2)", "y(3x^2-y^2)"),
        )
    ),
}


class OrbitalState(ArchiveSection):
    """
    A base section to define the orbital state of an atom.
    """

    # TODO: add the relativistic kappa_quantum_number
    n_quantum_number = Quantity(
        type=np.int32,
        description="""
        Principal quantum number n of the orbital state.
        """,
    )

    l_quantum_number = Quantity(
        type=np.int32,
        description="""
        Azimuthal quantum number l of the orbital state.
        """,
    )

    ml_quantum_number = Quantity(
        type=np.int32,
        description="""
        Azimuthal projection of the l vector.
        """,
    )

    j_quantum_number = Quantity(
        type=np.float64,
        shape=["1..2"],
        description="""
        Total angular momentum quantum number $j = |l-s| ... l+s$. Necessary with strong
        L-S coupling or non-collinear spin systems.
        """,
    )

    mj_quantum_number = Quantity(
        type=np.float64,
        shape=["*"],
        description="""
        Azimuthal projection of the $j$ vector. Necessary with strong L-S coupling or
        non-collinear spin systems.
        """,
    )

    ms_quantum_number = Quantity(
        type=np.int32,
        description="""
        Spin quantum number. Set to -1 for spin down and +1 for spin up. In non-collinear spin systems,
        the projection axis $z$ should also be defined.
        """,
    )

    degeneracy = Quantity(
        type=np.int32,
        description="""
        The number of states under the filling constraints applied to the orbital set.
        This implicitly assumes that all orbitals in the set are degenerate.
        """,
    )

    occupation = Quantity(
        type=np.int32,
        description="""
        The number of electrons in the orbital state. The state is fully occupied if
        occupation = degeneracy.
        """,
    )

    def multiplicity(self, quantum_number):
        return 2 * quantum_number + 1

    def set_degeneracy(self) -> int:
        """Set the degeneracy based on how specifically determined the quantum state is.
        This function can be triggered anytime to update the degeneracy."""
        # TODO: there are certain j (mj) specifications that may straddle just one l value
        if self.ml_quantum_number is not None:
            if self.ms_quantum_bool is not None:
                self.degeneracy = 1
            else:
                self.degeneracy = 2
        elif self.j_quantum_number is not None:
            degeneracy = 0
            for jj in self.j_quantum_number:
                if self.mj_quantum_number is not None:
                    mjs = RussellSaundersState.generate_MJs(
                        self.j_quantum_number[0], rising=True
                    )
                    degeneracy += len(
                        [mj for mj in mjs if mj in self.mj_quantum_number]
                    )
                else:
                    degeneracy += RussellSaundersState(jj, 1).degeneracy
            if self.ms_quantum_bool is not None:
                self.degeneracy = degeneracy / 2
            else:
                self.degeneracy = degeneracy
        elif self.l_quantum_number is not None:
            if self.ms_quantum_bool is not None:
                self.degeneracy = self.get_l_degeneracy() / 2
            else:
                self.degeneracy = self.get_l_degeneracy()
        return self.degeneracy

    def normalize(self, archive, logger):
        super().normalize(archive, logger)


# class Pseudopotential(ArchiveSection):
#     """
#     A base section to define the pseudopotential of an atom.
#     """

#     name = Quantity(
#         type=str,
#         shape=[],
#         description="""
#         Native code name of the pseudopotential.
#         """,
#     )

#     type = Quantity(
#         type=MEnum("US V", "US MBK", "PAW"),
#         shape=[],
#         description="""
#         Pseudopotential classification.
#         | abbreviation | description | DOI |
#         | ------------ | ----------- | --------- |
#         | `'US'`       | Ultra-soft  | |
#         | `'PAW'`      | Projector augmented wave | |
#         | `'V'`        | Vanderbilt | https://doi.org/10.1103/PhysRevB.47.6728 |
#         | `'MBK'`      | Morrison-Bylander-Kleinman | https://doi.org/10.1103/PhysRevB.41.7892 |
#         """,
#     )

#     norm_conserving = Quantity(
#         type=bool,
#         shape=[],
#         description="""
#         Denotes whether the pseudopotential is norm-conserving.
#         """,
#     )

#     cutoff = Quantity(
#         type=np.float64,
#         shape=[],
#         unit="joule",
#         description="""
#         Minimum recommended spherical cutoff energy for any plane-wave basis set
#         using the pseudopotential.
#         """,
#     )

#     xc_functional_name = Quantity(
#         type=str,
#         shape=["*"],
#         description="""
#         Name of the exchange-correlation functional used to generate the pseudopotential.
#         Follows the libxc naming convention.
#         """,
#     )

#     l_max = Quantity(
#         type=np.int32,
#         shape=[],
#         description="""
#         Maximum angular momentum of the pseudopotential projectors.
#         """,
#     )

#     lm_max = Quantity(
#         type=np.int32,
#         shape=[],
#         description="""
#         Maximum magnetic momentum of the pseudopotential projectors.
#         """,
#     )

#     def normalize(self, archive, logger):
#         super().normalize(archive, logger)


class CoreHole(ArchiveSection):
    """
    Describes the quantum state of a single hole in an open-shell core state. This is the physical interpretation.
    For modelling purposes, the electron charge excited may lie between 0 and 1. This follows a so-called Janak state.
    Sometimes, no electron is actually, excited, but just marked for excitation. This is denoted as an `initial` state.
    Any missing quantum numbers indicate some level of arbitrariness in the choice of the core hole, represented in the degeneracy.
    """

    # quantities: list[str] = SingleElectronState.quantities + ["n_electrons_excited"]

    n_electrons_excited = Quantity(
        type=np.float64,
        shape=[],
        description="""
        The electron charge excited for modelling purposes.
        Choices that deviate from 0 or 1 typically leverage Janak composition.
        Unless the `initial` state is chosen, the model corresponds to a single electron being excited in physical reality.
        """,
    )
    occupation = Quantity(
        type=np.float64,
        description="""
        The total number of electrons within the state (as defined by degeneracy)
        after exciting the model charge.
        """,
    )
    dscf_state = Quantity(
        type=MEnum("initial", "final"),
        shape=[],
        description="""
        The $\\Delta$-SCF state tag, used to identify the role in the workflow of the same name.
        Allowed values are `initial` (not to be confused with the _initial-state approximation_) and `final`.
        """,
    )

    def __setattr__(self, name, value):
        if name == "n_electrons_excited":
            if value < 0.0:
                raise ValueError("Number of excited electrons must be positive.")
            if value > 1.0:
                raise ValueError("Number of excited electrons must be less than 1.")
        elif name == "dscf_state":
            if value == "initial":
                self.n_electrons_excited = 0.0
                self.degeneracy = 1
        super().__setattr__(name, value)

    def _extract_orbital(self):
        """
        Gather atomic orbitals from `run.method.atom_parameters.core_hole`.
        Also apply normalization in the process.
        """
        # Collect the active orbitals from method
        methods = self.entry_archive.run[-1].method
        if not methods:
            return []
        atom_params = getattr(methods[-1], "atom_parameters", [])
        active_orbitals_run = []
        for param in atom_params:
            core_hole = param.core_hole
            if core_hole:
                active_orbitals_run.append(core_hole)
        if (
            len(active_orbitals_run) > 1
        ):  # FIXME: currently only one set of active orbitals is supported, remove for multiple
            self.logger.warn(
                """Multiple sets of active orbitals found.
                Currently, the topology only supports 1, so only the first set is used."""
            )
        # Map the active orbitals to the topology
        active_orbitals_results = []
        for active_orbitals in active_orbitals_run:
            active_orbitals.normalize(None, None)
            active_orbitals_new = CoreHole()
            for quantity_name in active_orbitals.quantities:
                setattr(
                    active_orbitals_new,
                    quantity_name,
                    getattr(active_orbitals, quantity_name),
                )
            active_orbitals_new.normalize(None, None)
            active_orbitals_results.append(active_orbitals_new)
            break  # FIXME: currently only one set of active orbitals is supported, remove for multiple
        return active_orbitals_results

    def set_occupation(self) -> float:
        """Set the occupation based on the number of excited electrons."""
        if not self.occupation:
            try:
                self.occupation = self.set_degeneracy() - self.n_electrons_excited
            except TypeError:
                raise AttributeError(
                    "Cannot set occupation without `n_electrons_excited`."
                )
        return self.occupation

    def normalize(self, archive, logger):
        super().normalize(archive, logger)
        # self.set_occupation()


class HubbardInteractions(ArchiveSection):
    """
    A base section to define the Hubbard interactions of the system.
    """

    orbital_ref = Quantity(
        type=OrbitalState,
        shape=["*"],
        description="""
        Reference to the OrbitalState sections that are used as a basis to obtain the Hubbard
        interaction matrices.
        """,
    )

    umn = Quantity(
        type=np.float64,
        shape=["*", "*"],
        unit="joule",
        description="""
        Value of the local Hubbard interaction matrix. The order of the rows and columns coincide
        with the elements in `orbital_ref`.
        """,
    )

    u = Quantity(
        type=np.float64,
        unit="joule",
        description="""
        Value of the (intraorbital) Hubbard interaction
        """,
        a_eln=ELNAnnotation(component="NumberEditQuantity"),
    )

    jh = Quantity(
        type=np.float64,
        unit="joule",
        description="""
        Value of the (interorbital) Hund's coupling.
        """,
        a_eln=ELNAnnotation(component="NumberEditQuantity"),
    )

    up = Quantity(
        type=np.float64,
        unit="joule",
        description="""
        Value of the (interorbital) Coulomb interaction. In rotational invariant systems, up = u - 2 * jh.
        """,
        a_eln=ELNAnnotation(component="NumberEditQuantity"),
    )

    j = Quantity(
        type=np.float64,
        unit="joule",
        description="""
        Value of the exchange interaction. In rotational invariant systems, j = jh.
        """,
        a_eln=ELNAnnotation(component="NumberEditQuantity"),
    )

    u_effective = Quantity(
        type=np.float64,
        unit="joule",
        description="""
        Value of the effective U parameter (u - j).
        """,
    )

    slater_integrals = Quantity(
        type=np.float64,
        shape=[3],
        unit="joule",
        description="""
        Value of the Slater integrals [F0, F2, F4] in spherical harmonics used to derive
        the local Hubbard interactions:

            u = ((2.0 / 7.0) ** 2) * (F0 + 5.0 * F2 + 9.0 * F4) / (4.0*np.pi)

            up = ((2.0 / 7.0) ** 2) * (F0 - 5.0 * F2 + 3.0 * 0.5 * F4) / (4.0*np.pi)

            jh = ((2.0 / 7.0) ** 2) * (5.0 * F2 + 15.0 * 0.25 * F4) / (4.0*np.pi)

        See e.g., Elbio Dagotto, Nanoscale Phase Separation and Colossal Magnetoresistance,
        Chapter 4, Springer Berlin (2003).
        """,
    )

    double_counting_correction = Quantity(
        type=str,
        description="""
        Name of the double counting correction algorithm applied.
        """,
        a_eln=ELNAnnotation(component="StringEditQuantity"),
    )

    def resolve_u_interactions(self, logger: BoundLogger) -> Optional[tuple]:
        """
        Resolves the Hubbard interactions (u, up, jh) from the Slater integrals (F0, F2, F4).

        Args:
            logger (BoundLogger): The logger to log messages.

        Returns:
            (Optional[tuple]): The Hubbard interactions (u, up, jh).
        """
        if self.slater_integrals is None or len(self.slater_integrals) == 3:
            logger.warning(
                "Could not find `HubbardInteractions.slater_integrals` or the length is not three."
            )
            return None
        f0 = self.slater_integrals[0]
        f2 = self.slater_integrals[1]
        f4 = self.slater_integrals[2]
        u_interaction = (
            ((2.0 / 7.0) ** 2)
            * (f0 + 5.0 * f2 + 9.0 * f4)
            / (4.0 * np.pi)
            * ureg("joule")
        )
        up_interaction = (
            ((2.0 / 7.0) ** 2)
            * (f0 - 5.0 * f2 + 3.0 * f4 / 2.0)
            / (4.0 * np.pi)
            * ureg("joule")
        )
        jh_interaction = (
            ((2.0 / 7.0) ** 2)
            * (5.0 * f2 + 15.0 * f4 / 4.0)
            / (4.0 * np.pi)
            * ureg("joule")
        )
        return u_interaction, up_interaction, jh_interaction

    def resolve_u_effective(self, logger: BoundLogger) -> Optional[np.float64]:
        """
        Resolves the effective U parameter (u - j).

        Args:
            logger (BoundLogger): The logger to log messages.

        Returns:
            (Optional[np.float64]): The effective U parameter.
        """
        if self.u is None or self.j is None:
            logger.warning(
                "Could not find `HubbardInteractions.u` or `HubbardInteractions.j`."
            )
            return None
        return self.u - self.j

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        # Obtain (u, up, jh) from slater_integrals
        if self.u is None and self.up is None and self.jh is None:
            self.u, self.up, self.jh = self.resolve_u_interactions(logger)

        # If u_effective is not available, calculate it
        if self.u_effective is None:
            self.u_effective = self.resolve_u_effective(logger)


class AtomsState(ArchiveSection):
    """
    A base section to define each atom state information.
    """

    # ? constraint to the normal chemical elements (no 'X' as defined in ASE included)
    chemical_symbol = Quantity(
        type=MEnum(ase.data.chemical_symbols[1:]),
        description="""
        Symbol of the element, e.g. 'H', 'Pb'. This quantity is equivalent to `atomic_numbers`.
        """,
    )

    atomic_number = Quantity(
        type=np.int32,
        description="""
        Atomic number Z. This quantity is equivalent to `chemical_symbol`.
        """,
    )

    orbitals = SubSection(sub_section=OrbitalState.m_def, repeats=True)

    charge = Quantity(
        type=np.int32,
        default=0,
        description="""
        Charge of the atom. It is defined as the number of extra electrons or holes in the
        atom. If the atom is neutral, charge = 0 and the summation of all (if available) the`OrbitalState.occupation`
        coincides with the `atomic_number`. Otherwise, charge can be any positive integer (+1, +2...)
        for cations or any negative integer (-1, -2...) for anions.

        Note: for `CoreHole` systems we do not consider the charge of the atom even if
        we do not store the final `OrbitalState` where the electron was excited to.
        """,
        a_eln=ELNAnnotation(component="NumberEditQuantity"),
    )

    # pseudopotential = SubSection(sub_section=Pseudopotential.m_def, repeats=False)

    core_hole = SubSection(sub_section=CoreHole.m_def, repeats=False)

    hubbard_interactions = SubSection(
        sub_section=HubbardInteractions.m_def, repeats=False
    )

    def resolve_chemical_symbol_and_number(self, logger: BoundLogger) -> None:
        """
        Resolves the chemical symbol from the atomic number and viceversa.

        Args:
            logger (BoundLogger): The logger to log messages.
        """
        if self.chemical_symbol is None and self.atomic_number is not None:
            try:
                self.chemical_symbol = ase.data.chemical_symbols[self.atomic_number]
            except IndexError:
                logger.error(
                    "The `AtomicState.atomic_number` is out of range of the periodic table."
                )
                return
        elif self.chemical_symbol is not None and self.atomic_number is None:
            try:
                self.atomic_number = ase.data.atomic_numbers[self.chemical_symbol]
            except IndexError:
                logger.error(
                    "The `AtomicState.chemical_symbol` is not recognized in the periodic table."
                )
                return

    def normalize(self, archive, logger):
        super().normalize(archive, logger)

        # Get chemical_symbol from atomic_number and viceversa
        self.resolve_chemical_symbol_and_number(logger)
