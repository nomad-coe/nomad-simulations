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
from typing import List, Union, Optional, Tuple

from nomad.units import ureg

from . import logger

from nomad_simulations.atoms_state import (
    OrbitalsState,
    CoreHole,
    HubbardInteractions,
    AtomsState,
)


class TestOrbitalsState:
    """
    Test the `OrbitalsState` class defined in atoms_state.py.
    """

    @staticmethod
    def add_state(
        orbital_state: OrbitalsState,
        l_number: int,
        ml_number: Optional[int],
        ms_number: Optional[float],
        j_number: Optional[List[float]],
        mj_number: Optional[List[float]],
    ) -> None:
        """Adds l and ml quantum numbers to the `OrbitalsState` section."""
        orbital_state.l_quantum_number = l_number
        orbital_state.ml_quantum_number = ml_number
        orbital_state.ms_quantum_number = ms_number
        orbital_state.j_quantum_number = j_number
        orbital_state.mj_quantum_number = mj_number

    @pytest.mark.parametrize(
        'number_label, values, results',
        [
            ('n_quantum_number', [-1, 0, 1, 2], [False, False, True, True]),
            ('l_quantum_number', [-2, 0, 1, 2], [False, False, True, True]),
            # l_quantum_number == 2 when testing 'ml_quantum_number'
            ('ml_quantum_number', [-3, 5, -2, 1], [False, False, True, True]),
            ('ms_quantum_number', [0, 10, -0.5, 0.5], [False, False, True, True]),
        ],
    )
    def test_check_quantum_numbers(
        self, number_label: str, values: List[int], results: List[bool]
    ):
        """
        Test the quantum number check for the `OrbitalsState` section.
        """
        orbital_state = OrbitalsState(n_quantum_number=2)
        for val, res in zip(values, results):
            if number_label == 'ml_quantum_number':
                orbital_state.l_quantum_number = 2
            setattr(orbital_state, number_label, val)
            check = orbital_state._check_quantum_numbers(logger)
            assert check == res

    @pytest.mark.parametrize(
        'quantum_name, value, expected_result',
        [
            ('l', 0, 's'),
            ('l', 1, 'p'),
            ('l', 2, 'd'),
            ('l', 3, 'f'),
            ('l', 4, None),
            ('ml', -1, 'x'),
            ('ml', 0, 'z'),
            ('ml', 1, 'y'),
            ('ml', -2, None),
            ('ms', -0.5, 'down'),
            ('ms', 0.5, 'up'),
            ('ms', -0.75, None),
            ('no_attribute', None, None),
        ],
    )
    def test_number_and_symbol(
        self,
        quantum_name: str,
        value: Union[int, float],
        expected_result: Optional[str],
    ):
        """
        Test the number and symbol resolution for each of the quantum numbers defined in the parametrization.
        """
        # Adding quantum numbers to the `OrbitalsState` section
        orbital_state = OrbitalsState(n_quantum_number=2)
        if quantum_name == 'ml':  # l_quantum_number must be specified
            orbital_state.l_quantum_number = 1
        setattr(orbital_state, f'{quantum_name}_quantum_number', value)

        # Making sure that the `'number'` is assigned
        resolved_type = orbital_state.resolve_number_and_symbol(
            quantum_name, 'number', logger
        )
        assert resolved_type == value

        # Resolving if the counter-type is assigned
        resolved_countertype = orbital_state.resolve_number_and_symbol(
            quantum_name, 'symbol', logger
        )
        assert resolved_countertype == expected_result

    @pytest.mark.parametrize(
        'l_quantum_number, ml_quantum_number, j_quantum_number, mj_quantum_number, ms_quantum_number, degeneracy',
        [
            (1, None, None, None, 0.5, 3),
            (1, None, None, None, None, 6),
            (1, -1, None, None, 0.5, 1),
            (1, -1, None, None, None, 2),
            # ! `RusselSanders` uses the electronic state, but here we are defining single orbitals. We need new methodology to treat these cases separately
            # ! and these tests for j_quantum_number and mj_quantum_number need to be updated.
            (1, -1, [1 / 2, 3 / 2], None, None, 6),
            (1, -1, [1 / 2, 3 / 2], [-3 / 2, 1 / 2, 1 / 2, 3 / 2], None, 2),
        ],
    )
    def test_degeneracy(
        self,
        l_quantum_number: int,
        ml_quantum_number: Optional[int],
        j_quantum_number: Optional[List[float]],
        mj_quantum_number: Optional[List[float]],
        ms_quantum_number: Optional[float],
        degeneracy: int,
    ):
        """
        Test the degeneracy of each orbital states defined in the parametrization.
        """
        orbital_state = OrbitalsState(n_quantum_number=2)
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

    def test_normalize(self):
        """
        Test the normalization of the `OrbitalsState`. Inputs are defined as the quantities of the `OrbitalsState` section.
        """
        orbital_state = OrbitalsState(n_quantum_number=2)
        self.add_state(orbital_state, 2, -2, None, None, None)
        orbital_state.normalize(None, logger)
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

    @pytest.mark.parametrize(
        'orbital_ref, degeneracy, n_excited_electrons, occupation',
        [
            (OrbitalsState(l_quantum_number=1), 6, 0.5, 5.5),
            (OrbitalsState(l_quantum_number=1, ml_quantum_number=-1), 2, 0.5, 1.5),
            (None, None, 0.5, None),
        ],
    )
    def test_occupation(
        self,
        orbital_ref: Optional[OrbitalsState],
        degeneracy: Optional[int],
        n_excited_electrons: float,
        occupation: Optional[float],
    ):
        """
        Test the occupation of a core hole for a given set of orbital reference and degeneracy.
        """
        core_hole = CoreHole(
            orbital_ref=orbital_ref, n_excited_electrons=n_excited_electrons
        )
        if orbital_ref is not None:
            assert orbital_ref.resolve_degeneracy() == degeneracy
        resolved_occupation = core_hole.resolve_occupation(logger)
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
        self,
        orbital_ref: Optional[OrbitalsState],
        n_excited_electrons: Optional[float],
        dscf_state: Optional[str],
        results: Tuple[Optional[float], Optional[float], Optional[float]],
    ):
        """
        Test the normalization of the `CoreHole`. Inputs are defined as the quantities of the `CoreHole` section.
        """
        core_hole = CoreHole(
            orbital_ref=orbital_ref,
            n_excited_electrons=n_excited_electrons,
            dscf_state=dscf_state,
        )
        core_hole.normalize(None, logger)
        assert core_hole.n_excited_electrons == results[0]
        if core_hole.orbital_ref:
            assert core_hole.orbital_ref.degeneracy == results[1]
            assert core_hole.orbital_ref.occupation == results[2]


class TestHubbardInteractions:
    """
    Test the `HubbardInteractions` class defined in atoms_state.py.
    """

    @pytest.mark.parametrize(
        'slater_integrals, results',
        [
            ([3.0, 2.0, 1.0], (0.1429146, -0.0357286, 0.0893216)),
            (None, (None, None, None)),
            ([3.0, 2.0, 1.0, 0.5], (None, None, None)),
        ],
    )
    def test_u_interactions(
        self,
        slater_integrals: Optional[List[float]],
        results: Tuple[Optional[float], Optional[float], Optional[float]],
    ):
        """
        Test the Hubbard interactions `U`, `U'`, and `J` for a given set of Slater integrals.
        """
        # Adding `slater_integrals` to the `HubbardInteractions` section
        hubbard_interactions = HubbardInteractions()
        if slater_integrals is not None:
            hubbard_interactions.slater_integrals = slater_integrals * ureg.eV

        # Resolving U, U', and J from class method
        (
            u_interaction,
            u_interorbital_interaction,
            j_hunds_coupling,
        ) = hubbard_interactions.resolve_u_interactions(logger)

        if None not in (u_interaction, u_interorbital_interaction, j_hunds_coupling):
            assert np.isclose(u_interaction.to('eV').magnitude, results[0])
            assert np.isclose(u_interorbital_interaction.to('eV').magnitude, results[1])
            assert np.isclose(j_hunds_coupling.to('eV').magnitude, results[2])
        else:
            assert np.isclose(u_interaction, results[0])
            assert np.isclose(u_interorbital_interaction, results[1])
            assert np.isclose(j_hunds_coupling, results[2])

    @pytest.mark.parametrize(
        'u_interaction, j_local_exchange_interaction, u_effective',
        [
            (3.0, 1.0, 2.0),
            (-3.0, 1.0, None),
            (3.0, None, 3.0),
            (None, 1.0, None),
        ],
    )
    def test_u_effective(
        self,
        u_interaction: Optional[float],
        j_local_exchange_interaction: Optional[float],
        u_effective: Optional[float],
    ):
        """
        Test the effective Hubbard interaction `U_eff` for a given set of Hubbard interactions `U` and `J`.
        """
        # Adding `u_interaction` and `j_local_exchange_interaction` to the `HubbardInteractions` section
        hubbard_interactions = HubbardInteractions()
        if u_interaction is not None:
            hubbard_interactions.u_interaction = u_interaction * ureg.eV
        if j_local_exchange_interaction is not None:
            hubbard_interactions.j_local_exchange_interaction = (
                j_local_exchange_interaction * ureg.eV
            )

        # Resolving Ueff from class method
        resolved_u_effective = hubbard_interactions.resolve_u_effective(logger)
        if resolved_u_effective is not None:
            assert np.isclose(resolved_u_effective.to('eV').magnitude, u_effective)
        else:
            assert np.isclose(resolved_u_effective, u_effective)

    def test_normalize(self):
        """
        Test the normalization of the `HubbardInteractions`. Inputs are defined as the quantities of the `HubbardInteractions` section.
        """
        # ? Is this enough for testing? Can we do more?
        hubbard_interactions = HubbardInteractions(
            u_interaction=3.0 * ureg.eV,
            u_interorbital_interaction=1.0 * ureg.eV,
            j_hunds_coupling=2.0 * ureg.eV,
            j_local_exchange_interaction=2.0 * ureg.eV,
        )
        hubbard_interactions.normalize(None, logger)
        assert np.isclose(hubbard_interactions.u_effective.to('eV').magnitude, 1.0)
        assert np.isclose(hubbard_interactions.u_interaction.to('eV').magnitude, 3.0)


class TestAtomsState:
    """
    Tests the `AtomsState` class defined in atoms_state.py.
    """

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
        self, chemical_symbol: str, atomic_number: int
    ):
        """
        Test the `chemical_symbol` and `atomic_number` resolution for the `AtomsState` section.
        """
        # Testing `chemical_symbol`
        atom_state = AtomsState(chemical_symbol=chemical_symbol)
        assert atom_state.resolve_atomic_number(logger) == atomic_number
        # Testing `atomic_number`
        atom_state.atomic_number = atomic_number
        assert atom_state.resolve_chemical_symbol(logger) == chemical_symbol
