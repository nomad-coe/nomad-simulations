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

from .utils import RussellSaundersState


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

orbitals_map = {
    "l_symbols": orbitals[-1],
    "ml_symbols": {i: orbitals[i] for i in range(4)},
    "ms_symbols": dict((zip((False, True), ("down", "up")))),
    "l_numbers": {v: k for k, v in orbitals[-1].items()},
    "ml_numbers": {k: {v: k for k, v in orbitals[k].items()} for k in range(4)},
    "ms_numbers": dict((zip((False, True), (-0.5, 0.5)))),
}


class OrbitalsState(ArchiveSection):
    """
    A base section used to define the orbital state of an atom.
    """

    # TODO: add the relativistic kappa_quantum_number
    n_quantum_number = Quantity(
        type=np.int32,
        description="""
        Principal quantum number of the orbital state.
        """,
    )

    l_quantum_number = Quantity(
        type=np.int32,
        description="""
        Azimuthal quantum number of the orbital state. This quantity is equivalent to `l_quantum_symbol`.
        """,
    )

    l_quantum_symbol = Quantity(
        type=np.int32,
        description="""
        Azimuthal quantum symbol of the orbital state, "s", "p", "d", "f", etc. This
        quantity is equivalent to `l_quantum_number`.
        """,
    )

    ml_quantum_number = Quantity(
        type=np.int32,
        description="""
        Azimuthal projection number of the `l` vector. This quantity is equivalent to `ml_quantum_symbol`.
        """,
    )

    ml_quantum_symbol = Quantity(
        type=np.int32,
        description="""
        Azimuthal projection symbol of the `l` vector, "x", "y", "z", etc. This quantity is equivalent
        to `ml_quantum_number`.
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
        Azimuthal projection of the `j` vector. Necessary with strong L-S coupling or
        non-collinear spin systems.
        """,
    )

    ms_quantum_number = Quantity(
        type=np.int32,
        description="""
        Spin quantum number. Set to -1 for spin down and +1 for spin up. In non-collinear spin
        systems, the projection axis $z$ should also be defined.
        """,
    )

    ms_quantum_symbol = Quantity(
        type=np.int32,
        description="""
        Spin quantum symbol. Set to 'down' for spin down and 'up' for spin up. In non-collinear
        spin systems, the projection axis $z$ should also be defined.
        """,
    )

    degeneracy = Quantity(
        type=np.int32,
        description="""
        The degeneracy of the orbital state. The degeneracy is the number of states with the same
        energy. It is equal to 2 * l + 1 for non-relativistic systems and 2 * j + 1 for
        relativistic systems, if ms_quantum_number is defined (otherwise a factor of 2 is included).
        """,
    )

    occupation = Quantity(
        type=np.float64,
        description="""
        The number of electrons in the orbital state. The state is fully occupied if
        occupation = degeneracy.
        """,
    )

    def resolve_number_and_symbol(
        self, quantum_number: str, type: str, logger: BoundLogger
    ) -> None:
        """
        Resolves the quantum number from the `orbitals_map` on the passed type. `type` can be either
        'number' or 'symbol'. If the quantum type is not found, then the countertype
        (e.g., type = 'number' => countertype = 'symbol') is used to resolve it.

        Args:
            quantum_number (str): The quantum number to resolve. Can be 'l', 'ml' or 'ms'.
            type (str): The type of the quantum number. Can be 'number' or 'symbol'.
            logger (BoundLogger): The logger to log messages.
        """
        if quantum_number not in ["l", "ml", "ms"]:
            logger.warning(
                "The quantum number is not recognized. Try 'l', 'ml' or 'ms'."
            )
            return
        if type not in ["number", "symbol"]:
            logger.warning("The type is not recognized. Try 'number' or 'symbol'.")
            return

        _countertype_map = {
            "number": "symbol",
            "symbol": "number",
        }
        # Check if quantity already exists
        quantity = getattr(self, f"{quantum_number}_quantum_{type}")
        if quantity:
            return

        # If not, check whether the countertype exists
        other_quantity = getattr(
            self, f"{quantum_number}_quantum_{_countertype_map[type]}"
        )
        if not other_quantity:
            logger.warning(
                f"Could not find the {quantum_number}_quantum_{type} countertype {_countertype_map[type]}."
            )
            return

        # If the counterpart exists, then resolve the quantity from the orbitals_map
        quantity = getattr(orbitals_map, f"{quantum_number}_{type}s")
        setattr(self, f"{quantum_number}_quantum_{type}", quantity)

    def resolve_degeneracy(self) -> Optional[int]:
        """
        Resolves the degeneracy of the orbital state. If `j_quantum_number` is not defined, then the
        degeneracy is computed from the `l_quantum_number` and `ml_quantum_number`. If `j_quantum_number`
        is defined, then the degeneracy is computed from the `j_quantum_number` and `mj_quantum_number`.
        There is a factor of 2 included if `ms_quantum_number` is not defined (for orbitals which
        are spin-degenerated).

        Returns:
            (Optional[int]): The degeneracy of the orbital state.
        """
        if self.l_quantum_number and self.ml_quantum_number is None:
            if self.ms_quantum_number:
                degeneracy = 2 * self.l_quantum_number + 1
            else:
                degeneracy = 2 * (2 * self.l_quantum_number + 1)
        elif self.l_quantum_number and self.ml_quantum_number:
            if self.ms_quantum_number:
                degeneracy = 1
            else:
                degeneracy = 2
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
        return degeneracy

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        # Resolving the quantum numbers and symbols if not available
        for quantum_number in ["l", "ml", "ms"]:
            for type in ["number", "symbol"]:
                self.resolve_number_and_symbol(quantum_number, type, logger)

        # Resolve the degeneracy
        self.degeneracy = self.resolve_degeneracy()


class CoreHole(ArchiveSection):
    """
    A base section used to define the core-hole state of an atom by referencing the `OrbitalsState`
    section where the core-hole was generated.
    """

    orbital_ref = Quantity(
        type=OrbitalsState,
        description="""
        Reference to the OrbitalsState section that is used as a basis to obtain the `CoreHole` section.
        """,
        a_eln=ELNAnnotation(component="ReferenceEditQuantity"),
    )

    n_excited_electrons = Quantity(
        type=np.float64,
        description="""
        The electron charge excited for modelling purposes. This is a number between 0 and 1 (Janak state).
        If `dscf_state` is set to 'initial', then this quantity is set to None (but assumed to be excited
        to an excited state).
        """,
    )

    dscf_state = Quantity(
        type=MEnum("initial", "final"),
        description="""
        Tag used to identify the role in the workflow of the same name. Allowed values are 'initial'
        (not to be confused with the _initial-state approximation_) and 'final'. If 'initial'
        is used, then `n_excited_electrons` is set to None and the `orbital_ref.degeneracy` is
        set to 1.
        """,
    )

    def resolve_occupation(self, logger: BoundLogger) -> None:
        """
        Resolves the occupation of the orbital state. The occupation is resolved from the degeneracy
        and the number of excited electrons.

        Args:
            logger (BoundLogger): The logger to log messages.
        """
        if self.orbital_ref is None or self.n_excited_electrons is None:
            logger.warning(
                "Cannot resolve occupation without `orbital_ref` or `n_excited_electrons`."
            )
            return
        if self.orbital_ref.occupation is None:
            degeneracy = self.orbital_ref.resolve_degeneracy()
            if degeneracy is None:
                logger.warning("Cannot resolve occupation without `degeneracy`.")
                return
            self.orbital_ref.occupation = degeneracy - self.n_excited_electrons

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        # Check if n_excited_electrons is between 0 and 1
        if 0.0 <= self.n_excited_electrons <= 1.0:
            logger.error("Number of excited electrons must be between 0 and 1.")

        # If dscf_state is 'initial', then n_excited_electrons is set to 0
        if self.dscf_state == "initial":
            self.n_excited_electrons = None
            self.degeneracy = 1

        # Resolve the occupation of the active orbital state
        if self.orbital_ref is not None and self.n_excited_electrons:
            if self.orbital_ref.occupation is None:
                self.resolve_occupation(logger)


class HubbardInteractions(ArchiveSection):
    """
    A base section to define the Hubbard interactions of the system.
    """

    orbitals_ref = Quantity(
        type=OrbitalsState,
        shape=["*"],
        description="""
        Reference to the `OrbitalsState` sections that are used as a basis to obtain the Hubbard
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
                "Could not find `slater_integrals` or the length is not three."
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

    def normalize(self, archive, logger) -> None:
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

    orbitals_state = SubSection(sub_section=OrbitalsState.m_def, repeats=True)

    charge = Quantity(
        type=np.int32,
        default=0,
        description="""
        Charge of the atom. It is defined as the number of extra electrons or holes in the
        atom. If the atom is neutral, charge = 0 and the summation of all (if available) the`OrbitalsState.occupation`
        coincides with the `atomic_number`. Otherwise, charge can be any positive integer (+1, +2...)
        for cations or any negative integer (-1, -2...) for anions.

        Note: for `CoreHole` systems we do not consider the charge of the atom even if
        we do not store the final `OrbitalsState` where the electron was excited to.
        """,
        a_eln=ELNAnnotation(component="NumberEditQuantity"),
    )

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
                    "The `AtomsState.atomic_number` is out of range of the periodic table."
                )
                return
        elif self.chemical_symbol is not None and self.atomic_number is None:
            try:
                self.atomic_number = ase.data.atomic_numbers[self.chemical_symbol]
            except IndexError:
                logger.error(
                    "The `AtomsState.chemical_symbol` is not recognized in the periodic table."
                )
                return

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        # Get chemical_symbol from atomic_number and viceversa
        self.resolve_chemical_symbol_and_number(logger)
