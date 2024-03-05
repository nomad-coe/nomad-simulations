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

import pytest
import numpy as np
import logging

from nomad.units import ureg

from simulationdataschema.atoms_state import (
    OrbitalsState,
    CoreHole,
    HubbardInteractions,
    AtomsState,
)


class TestOrbitalsState:
    """
    Test the `OrbitalsState` class defined in atoms_state.py.
    """

    logger = logging.getLogger(__name__)

    @staticmethod
    def add_quantum_numbers(orbital_state, quantum_name, quantum_type, value) -> None:
        """Adds quantum numbers to the `OrbitalsState` section."""
        if quantum_name == 'ml':  # l_quantum_number must be specified
            orbital_state.l_quantum_number = 1
        setattr(orbital_state, f'{quantum_name}_quantum_{quantum_type}', value)

    @staticmethod
    def add_state(
        orbital_state, l_number, ml_number, ms_number, j_number, mj_number
    ) -> None:
        """Adds l and ml quantum numbers to the `OrbitalsState` section."""
        orbital_state.l_quantum_number = l_number
        orbital_state.ml_quantum_number = ml_number
        orbital_state.ms_quantum_number = ms_number
        orbital_state.j_quantum_number = j_number
        orbital_state.mj_quantum_number = mj_number

    @pytest.fixture(autouse=True)
    def orbital_state(self) -> OrbitalsState:
        return OrbitalsState(n_quantum_number=2)

    @pytest.mark.parametrize(
        'number, values, results',
        [
            ('n_quantum_number', [-1, 0, 1, 2], [False, False, True, True]),
            ('l_quantum_number', [-2, 0, 1, 2], [False, False, True, True]),
            # l_quantum_number == 2 when testing 'ml_quantum_number'
            ('ml_quantum_number', [-3, 5, -2, 1], [False, False, True, True]),
            ('ms_quantum_number', [0, 10, -0.5, 0.5], [False, False, True, True]),
        ],
    )
    def test_check_quantum_numbers(self, orbital_state, number, values, results):
        """
        Test the quantum number check for the `OrbitalsState` section.
        """
        for val, res in zip(values, results):
            if number == 'ml_quantum_number':
                orbital_state.l_quantum_number = 2
            setattr(orbital_state, number, val)
            check = orbital_state._check_quantum_numbers(self.logger)
            assert check == res

    @pytest.mark.parametrize(
        'quantum_name, quantum_type, value, countertype, expected_result',
        [
            ('l', 'number', 0, 'symbol', 's'),
            ('l', 'number', 1, 'symbol', 'p'),
            ('l', 'number', 2, 'symbol', 'd'),
            ('l', 'number', 3, 'symbol', 'f'),
            ('l', 'number', 4, 'symbol', None),
            ('ml', 'number', -1, 'symbol', 'x'),
            ('ml', 'number', 0, 'symbol', 'z'),
            ('ml', 'number', 1, 'symbol', 'y'),
            ('ml', 'number', -2, 'symbol', None),
            ('ms', 'number', -0.5, 'symbol', 'down'),
            ('ms', 'number', 0.5, 'symbol', 'up'),
            ('ms', 'number', -0.75, 'symbol', None),
            ('no_attribute', 'number', None, 'symbol', None),
        ],
    )
    def test_number_and_symbol(
        self,
        orbital_state,
        quantum_name,
        quantum_type,
        value,
        countertype,
        expected_result,
    ):
        """
        Test the number and symbol resolution for each of the quantum numbers defined in the parametrization.
        """
        # Adding quantum numbers to the `OrbitalsState` section
        self.add_quantum_numbers(orbital_state, quantum_name, quantum_type, value)

        # Making sure that the `quantum_type` is assigned
        resolved_type = orbital_state.resolve_number_and_symbol(
            quantum_name, quantum_type, self.logger
        )
        assert resolved_type == value

        # Resolving if the counter-type is assigned
        resolved_countertype = orbital_state.resolve_number_and_symbol(
            quantum_name, countertype, self.logger
        )
        assert resolved_countertype == expected_result

    @pytest.mark.parametrize(
        'l_quantum_number, ml_quantum_number, j_quantum_number, mj_quantum_number, ms_quantum_number, degeneracy',
        [
            (1, None, None, None, 0.5, 3),
            (1, None, None, None, None, 6),
            (1, -1, None, None, 0.5, 1),
            (1, -1, None, None, None, 2),
            # ! these J and MJ tests are unphysical; check what is going on
            (1, -1, [1 / 2, 3 / 2], None, None, 6),
            (1, -1, [1 / 2, 3 / 2], [-3 / 2, 1 / 2, 1 / 2, 3 / 2], None, 2),
        ],
    )
    def test_degeneracy(
        self,
        orbital_state,
        l_quantum_number,
        ml_quantum_number,
        j_quantum_number,
        mj_quantum_number,
        ms_quantum_number,
        degeneracy,
    ):
        """
        Test the degeneracy of each of orbital states defined in the parametrization.
        """
        self.add_state(
            orbital_state,
            l_quantum_number,
            ml_quantum_number,
            ms_quantum_number,
            j_quantum_number,
            mj_quantum_number,
        )
        resolved_degeneracy = orbital_state.resolve_degeneracy()
        assert resolved_degeneracy == degeneracy

    def test_normalize(self, orbital_state):
        """
        Test the normalization of the `OrbitalsState`. Inputs are defined as the quantities of the `OrbitalsState` section.
        """
        self.add_state(orbital_state, 2, -2, None, None, None)
        orbital_state.normalize(None, self.logger)
        assert orbital_state.n_quantum_number == 2
        assert orbital_state.l_quantum_number == 2
        assert orbital_state.l_quantum_symbol == 'd'
        assert orbital_state.ml_quantum_number == -2
        assert orbital_state.ml_quantum_symbol == 'xy'
        assert orbital_state.degeneracy == 2


class TestCoreHole:
    """
    Test the `CoreHole` class defined in atoms_state.py.
    """

    logger = logging.getLogger(__name__)

    @pytest.fixture(autouse=True)
    def core_hole(self) -> CoreHole:
        return CoreHole()

    @pytest.mark.parametrize(
        'orbital_ref, degeneracy, n_excited_electrons, occupation',
        [
            (OrbitalsState(l_quantum_number=1), 6, 0.5, 5.5),
            (OrbitalsState(l_quantum_number=1, ml_quantum_number=-1), 2, 0.5, 1.5),
            (None, None, 0.5, None),
        ],
    )
    def test_occupation(
        self, core_hole, orbital_ref, degeneracy, n_excited_electrons, occupation
    ):
        """
        Test the occupation of a core hole for a given set of orbital reference and degeneracy.
        """
        core_hole.orbital_ref = orbital_ref
        if orbital_ref is not None:
            assert orbital_ref.resolve_degeneracy() == degeneracy
        core_hole.n_excited_electrons = n_excited_electrons
        resolved_occupation = core_hole.resolve_occupation(self.logger)
        if resolved_occupation is not None:
            assert np.isclose(resolved_occupation, occupation)
        else:
            assert resolved_occupation == occupation

    @pytest.mark.parametrize(
        'orbital_ref, n_excited_electrons, dscf_state, results',
        [
            (OrbitalsState(l_quantum_number=1), -0.5, None, (-0.5, None, None)),
            (OrbitalsState(l_quantum_number=1), 0.5, None, (0.5, None, 5.5)),
            (
                OrbitalsState(l_quantum_number=1, ml_quantum_number=-1),
                0.5,
                None,
                (0.5, None, 1.5),
            ),
            (OrbitalsState(l_quantum_number=1), 0.5, 'initial', (None, 1, None)),
            (OrbitalsState(l_quantum_number=1), 0.5, 'final', (0.5, None, 5.5)),
            (None, 0.5, None, (0.5, None, None)),
        ],
    )
    def test_normalize(
        self, core_hole, orbital_ref, n_excited_electrons, dscf_state, results
    ):
        """
        Test the normalization of the `CoreHole`. Inputs are defined as the quantities of the `CoreHole` section.
        """
        core_hole.orbital_ref = orbital_ref
        core_hole.n_excited_electrons = n_excited_electrons
        core_hole.dscf_state = dscf_state

        core_hole.normalize(None, self.logger)

        assert core_hole.n_excited_electrons == results[0]
        if core_hole.orbital_ref:
            assert core_hole.orbital_ref.degeneracy == results[1]
            assert core_hole.orbital_ref.occupation == results[2]


class TestHubbardInteractions:
    """
    Test the `HubbardInteractions` class defined in atoms_state.py.
    """

    logger = logging.getLogger(__name__)

    @staticmethod
    def add_slater_interactions(hubbard_interactions, slater_integrals) -> None:
        """Adds `slater_integrals` (in eV) to the `HubbardInteractions` section."""
        if slater_integrals is not None:
            hubbard_interactions.slater_integrals = slater_integrals * ureg('eV')

    @staticmethod
    def add_u_j(hubbard_interactions, u, j) -> None:
        """Adds `u_interaction` and `j_local_exchange_interaction` (in eV) to the `HubbardInteractions` section."""
        if u is not None:
            hubbard_interactions.u_interaction = u * ureg('eV')
        if j is not None:
            hubbard_interactions.j_local_exchange_interaction = j * ureg('eV')

    @pytest.fixture(autouse=True)
    def hubbard_interactions(self) -> HubbardInteractions:
        return HubbardInteractions()

    @pytest.mark.parametrize(
        'slater_integrals, results',
        [
            ([3.0, 2.0, 1.0], (0.1429146, -0.0357286, 0.0893216)),
            (None, (None, None, None)),
            ([3.0, 2.0, 1.0, 0.5], (None, None, None)),
        ],
    )
    def test_u_interactions(self, hubbard_interactions, slater_integrals, results):
        """
        Test the Hubbard interactions `U`, `U'`, and `J` for a given set of Slater integrals.
        """
        # Adding `slater_integrals` to the `HubbardInteractions` section
        self.add_slater_interactions(hubbard_interactions, slater_integrals)

        # Resolving U, U', and J from class method
        (
            u_interaction,
            u_interorbital_interaction,
            j_hunds_coupling,
        ) = hubbard_interactions.resolve_u_interactions(self.logger)

        if None not in (u_interaction, u_interorbital_interaction, j_hunds_coupling):
            assert np.isclose(u_interaction.to('eV').magnitude, results[0])
            assert np.isclose(u_interorbital_interaction.to('eV').magnitude, results[1])
            assert np.isclose(j_hunds_coupling.to('eV').magnitude, results[2])
        else:
            assert (
                u_interaction,
                u_interorbital_interaction,
                j_hunds_coupling,
            ) == results

    @pytest.mark.parametrize(
        'u_interaction, j_local_exchange_interaction, u_effective',
        [
            (3.0, 1.0, 2.0),
            (-3.0, 1.0, -4.0),
            (3.0, None, None),
            (None, 1.0, None),
        ],
    )
    def test_u_effective(
        self,
        hubbard_interactions,
        u_interaction,
        j_local_exchange_interaction,
        u_effective,
    ):
        """
        Test the effective Hubbard interaction `Ueff` for a given set of Hubbard interactions `U` and `J`.
        """
        # Adding `u_interaction` and `j_local_exchange_interaction` to the `HubbardInteractions` section
        self.add_u_j(hubbard_interactions, u_interaction, j_local_exchange_interaction)

        # Resolving Ueff from class method
        resolved_u_effective = hubbard_interactions.resolve_u_effective(self.logger)
        if resolved_u_effective is not None:
            assert np.isclose(resolved_u_effective.to('eV').magnitude, u_effective)
        else:
            assert resolved_u_effective == u_effective

    def test_normalize(self, hubbard_interactions):
        """
        Test the normalization of the `HubbardInteractions`. Inputs are defined as the quantities of the `HubbardInteractions` section.
        """
        # ? Is this enough for testing? Can we do more?
        self.add_u_j(hubbard_interactions, 3.0, 2.0)
        hubbard_interactions.u_interorbital_interaction = 1.0 * ureg('eV')
        hubbard_interactions.j_hunds_coupling = 2.0 * ureg('eV')

        hubbard_interactions.normalize(None, self.logger)
        assert np.isclose(hubbard_interactions.u_effective.to('eV').magnitude, 1.0)
        assert np.isclose(hubbard_interactions.u_interaction.to('eV').magnitude, 3.0)
        assert hubbard_interactions.slater_integrals is None


class TestAtomsState:
    """
    Tests the `AtomsState` class defined in atoms_state.py.
    """

    logger = logging.getLogger(__name__)

    @staticmethod
    def add_element_information(atom_state, quantity_name, value) -> None:
        setattr(atom_state, quantity_name, value)

    @pytest.fixture(autouse=True)
    def atom_state(self) -> AtomsState:
        return AtomsState()

    @pytest.mark.parametrize(
        'chemical_symbol, atomic_number',
        [
            ('Fe', 26),
            ('H', 1),
            ('Cu', 29),
            ('O', 8),
        ],
    )
    def test_chemical_symbol_and_atomic_number(
        self, atom_state, chemical_symbol, atomic_number
    ):
        """
        Test the `chemical_symbol` and `atomic_number` resolution for the `AtomsState` section.
        """
        # Testing `chemical_symbol`
        self.add_element_information(atom_state, 'chemical_symbol', chemical_symbol)
        resolved_atomic_number = atom_state.resolve_atomic_number(self.logger)
        assert resolved_atomic_number == atomic_number
        # Testing `atomic_number`
        self.add_element_information(atom_state, 'atomic_number', atomic_number)
        resolved_chemical_symbol = atom_state.resolve_chemical_symbol(self.logger)
        assert resolved_chemical_symbol == chemical_symbol
