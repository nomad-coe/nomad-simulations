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

from typing import Optional

import pytest
from nomad.datamodel import EntryArchive

from nomad_simulations.schema_packages.atoms_state import AtomsState, OrbitalsState
from nomad_simulations.schema_packages.general import Simulation
from nomad_simulations.schema_packages.model_method import (
    TB,
    SlaterKoster,
    SlaterKosterBond,
    Wannier,
)
from nomad_simulations.schema_packages.model_system import AtomicCell, ModelSystem

from . import logger
from .conftest import generate_simulation


class TestTB:
    """
    Test the `TB` class defined in `model_method.py`.
    """

    @pytest.mark.parametrize(
        'tb_section, result',
        [(Wannier(), 'Wannier'), (SlaterKoster(), 'SlaterKoster'), (TB(), None)],
    )
    def test_resolve_type(self, tb_section: TB, result: Optional[str]):
        """
        Test the `resolve_type` method.

        Args:
            tb_section (TB): The TB section to resolve the type from.
            result (Optional[str]): The expected type of the TB section.
        """
        assert tb_section.resolve_type() == result

    @pytest.mark.parametrize(
        'model_systems, model_index, result',
        [
            # no `ModelSystem` sections
            ([], 0, None),
            # `model_index` out of range
            ([ModelSystem()], 1, None),
            # no `is_representative` in `ModelSystem`
            ([ModelSystem(is_representative=False)], 0, None),
            # no `cell` section in `ModelSystem`
            ([ModelSystem(is_representative=True)], 0, None),
            # no `AtomsState` in `AtomicCell`
            ([ModelSystem(is_representative=True, cell=[AtomicCell()])], 0, None),
            # no `model_system` child section under `ModelSystem`
            (
                [
                    ModelSystem(
                        is_representative=True,
                        cell=[AtomicCell(atoms_state=[AtomsState()])],
                    )
                ],
                0,
                None,
            ),
            # `type` for the `model_system` child is not `'active_atom'`
            (
                [
                    ModelSystem(
                        is_representative=True,
                        cell=[AtomicCell(atoms_state=[AtomsState()])],
                        model_system=[ModelSystem(type='bulk')],
                    )
                ],
                0,
                [],
            ),
            # wrong index for `AtomsState in active atom
            (
                [
                    ModelSystem(
                        is_representative=True,
                        cell=[AtomicCell(atoms_state=[AtomsState()])],
                        model_system=[
                            ModelSystem(type='active_atom', atom_indices=[2])
                        ],
                    )
                ],
                0,
                [],
            ),
            # empty `OrbitalsState` in `AtomsState`
            (
                [
                    ModelSystem(
                        is_representative=True,
                        cell=[AtomicCell(atoms_state=[AtomsState(orbitals_state=[])])],
                        model_system=[
                            ModelSystem(type='active_atom', atom_indices=[0])
                        ],
                    )
                ],
                0,
                [],
            ),
            # valid case
            (
                [
                    ModelSystem(
                        is_representative=True,
                        cell=[
                            AtomicCell(
                                atoms_state=[
                                    AtomsState(
                                        orbitals_state=[
                                            OrbitalsState(l_quantum_symbol='s')
                                        ]
                                    )
                                ]
                            )
                        ],
                        model_system=[
                            ModelSystem(type='active_atom', atom_indices=[0])
                        ],
                    )
                ],
                0,
                [OrbitalsState(l_quantum_symbol='s')],
            ),
        ],
    )
    def test_resolve_orbital_references(
        self,
        model_systems: Optional[list[ModelSystem]],
        model_index: int,
        result: Optional[list[OrbitalsState]],
    ):
        """
        Test the `resolve_orbital_references` method.

        Args:
            model_systems (Optional[list[ModelSystem]]): The `model_system` section to add to `Simulation`.
            model_index (int): The index of the `ModelSystem` section to resolve the orbital references from.
            result (Optional[list[OrbitalsState]]): The expected orbital references.
        """
        tb_method = TB()
        simulation = generate_simulation(model_method=tb_method)
        simulation.model_system = model_systems
        orbitals_ref = tb_method.resolve_orbital_references(
            model_systems=model_systems,
            logger=logger,
            model_index=model_index,
        )
        if not orbitals_ref:
            assert orbitals_ref == result
        else:
            assert orbitals_ref[0].l_quantum_symbol == result[0].l_quantum_symbol

    @pytest.mark.parametrize(
        'tb_section, result_type, model_systems, result',
        [
            # no method `type` extracted
            (TB(), 'unavailable', [], None),
            # method `type` extracted
            (Wannier(), 'Wannier', [], None),
            # no `ModelSystem` sections
            (Wannier(), 'Wannier', [], None),
            # no `is_representative` in `ModelSystem`
            (Wannier(), 'Wannier', [ModelSystem(is_representative=False)], None),
            # no `cell` section in `ModelSystem`
            (Wannier(), 'Wannier', [ModelSystem(is_representative=True)], None),
            # no `AtomsState` in `AtomicCell`
            (
                Wannier(),
                'Wannier',
                [ModelSystem(is_representative=True, cell=[AtomicCell()])],
                None,
            ),
            # no `model_system` child section under `ModelSystem`
            (
                Wannier(),
                'Wannier',
                [
                    ModelSystem(
                        is_representative=True,
                        cell=[AtomicCell(atoms_state=[AtomsState()])],
                    )
                ],
                None,
            ),
            # `type` for the `model_system` child is not `'active_atom'`
            (
                Wannier(),
                'Wannier',
                [
                    ModelSystem(
                        is_representative=True,
                        cell=[AtomicCell(atoms_state=[AtomsState()])],
                        model_system=[ModelSystem(type='bulk')],
                    )
                ],
                None,
            ),
            # wrong index for `AtomsState in active atom
            (
                Wannier(),
                'Wannier',
                [
                    ModelSystem(
                        is_representative=True,
                        cell=[AtomicCell(atoms_state=[AtomsState()])],
                        model_system=[
                            ModelSystem(type='active_atom', atom_indices=[2])
                        ],
                    )
                ],
                None,
            ),
            # empty `OrbitalsState` in `AtomsState`
            (
                Wannier(),
                'Wannier',
                [
                    ModelSystem(
                        is_representative=True,
                        cell=[AtomicCell(atoms_state=[AtomsState(orbitals_state=[])])],
                        model_system=[
                            ModelSystem(type='active_atom', atom_indices=[0])
                        ],
                    )
                ],
                None,
            ),
            # `Wannier.orbitals_ref` already set up
            (
                Wannier(orbitals_ref=[OrbitalsState(l_quantum_symbol='d')]),
                'Wannier',
                [
                    ModelSystem(
                        is_representative=True,
                        cell=[
                            AtomicCell(
                                atoms_state=[
                                    AtomsState(
                                        orbitals_state=[
                                            OrbitalsState(l_quantum_symbol='s')
                                        ]
                                    )
                                ]
                            )
                        ],
                        model_system=[
                            ModelSystem(type='active_atom', atom_indices=[0])
                        ],
                    )
                ],
                [OrbitalsState(l_quantum_symbol='d')],
            ),
            # valid case
            (
                Wannier(),
                'Wannier',
                [
                    ModelSystem(
                        is_representative=True,
                        cell=[
                            AtomicCell(
                                atoms_state=[
                                    AtomsState(
                                        orbitals_state=[
                                            OrbitalsState(l_quantum_symbol='s')
                                        ]
                                    )
                                ]
                            )
                        ],
                        model_system=[
                            ModelSystem(type='active_atom', atom_indices=[0])
                        ],
                    )
                ],
                [OrbitalsState(l_quantum_symbol='s')],
            ),
        ],
    )
    def test_normalize(
        self,
        tb_section: TB,
        result_type: Optional[str],
        model_systems: Optional[list[ModelSystem]],
        result: Optional[list[OrbitalsState]],
    ):
        """
        Test the `resolve_orbital_references` method.

        Args:
            tb_section (TB): The TB section to resolve the type from.
            result_type (Optional[str]): The expected type of the TB section.
            model_systems (Optional[list[ModelSystem]]): The `model_system` section to add to `Simulation`.
            result (Optional[list[OrbitalsState]]): The expected orbital references.
        """
        simulation = generate_simulation(model_method=tb_section)
        simulation.model_system = model_systems
        tb_section.normalize(EntryArchive(), logger)
        assert tb_section.type == result_type
        if tb_section.orbitals_ref is not None:
            assert len(tb_section.orbitals_ref) == 1
            assert (
                tb_section.orbitals_ref[0].l_quantum_symbol
                == result[0].l_quantum_symbol
            )
        else:
            assert tb_section.orbitals_ref == result


class TestWannier:
    """
    Test the `Wannier` class defined in `model_method.py`.
    """

    @pytest.mark.parametrize(
        'localization_type, is_maximally_localized, result_localization_type',
        [
            # `localization_type` and `is_maximally_localized` are `None`
            (None, None, None),
            # `localization_type` set while `is_maximally_localized` is `None`
            ('single_shot', None, 'single_shot'),
            # normalizing from `is_maximally_localized`
            (None, True, 'maximally_localized'),
            (None, False, 'single_shot'),
        ],
    )
    def test_normalize(
        self,
        localization_type: Optional[str],
        is_maximally_localized: bool,
        result_localization_type: Optional[str],
    ):
        """
        Test the `normalize` method .

        Args:
            localization_type (Optional[str]): The localization type.
            is_maximally_localized (bool): If the localization is maximally-localized or a single-shot.
            result_localization_type (Optional[str]): The expected `localization_type` after normalization.
        """
        wannier = Wannier(
            localization_type=localization_type,
            is_maximally_localized=is_maximally_localized,
        )
        wannier.normalize(EntryArchive(), logger)
        assert wannier.localization_type == result_localization_type


class TestSlaterKosterBond:
    """
    Test the `SlaterKosterBond` class defined in `model_method.py`.
    """

    @pytest.mark.parametrize(
        'orbital_1, orbital_2, bravais_vector, result',
        [
            # no `OrbitalsState` sections
            (None, None, (), None),
            (None, OrbitalsState(), (), None),
            (OrbitalsState(), None, (), None),
            # no `bravais_vector`
            (OrbitalsState(), OrbitalsState(), None, None),
            # no `l_quantum_symbol` in `OrbitalsState`
            (OrbitalsState(), OrbitalsState(), (0, 0, 0), None),
            # valid cases
            (
                OrbitalsState(l_quantum_symbol='s'),
                OrbitalsState(l_quantum_symbol='s'),
                (0, 0, 0),
                'sss',
            ),
            (
                OrbitalsState(l_quantum_symbol='s'),
                OrbitalsState(l_quantum_symbol='p'),
                (0, 0, 0),
                'sps',
            ),
        ],
    )
    def test_resolve_bond_name_from_references(
        self,
        orbital_1: Optional[OrbitalsState],
        orbital_2: Optional[OrbitalsState],
        bravais_vector: Optional[tuple],
        result: Optional[str],
    ):
        """
        Test the `resolve_bond_name_from_references` method.

        Args:
            orbital_1 (Optional[OrbitalsState]): The first `OrbitalsState` section.
            orbital_2 (Optional[OrbitalsState]): The second `OrbitalsState` section.
            bravais_vector (Optional[tuple]): The bravais vector.
            result (Optional[str]): The expected bond name.
        """
        sk_bond = SlaterKosterBond()
        bond_name = sk_bond.resolve_bond_name_from_references(
            orbital_1=orbital_1,
            orbital_2=orbital_2,
            bravais_vector=bravais_vector,
            logger=logger,
        )
        assert bond_name == result

    @pytest.mark.parametrize(
        'orbital_1, orbital_2, bravais_vector, result',
        [
            # no `OrbitalsState` sections
            (None, None, [], None),
            (None, OrbitalsState(), [], None),
            (OrbitalsState(), None, [], None),
            # no `bravais_vector`
            (OrbitalsState(), OrbitalsState(), None, None),
            # no `l_quantum_symbol` in `OrbitalsState`
            (OrbitalsState(), OrbitalsState(), (0, 0, 0), None),
            # valid cases
            (
                OrbitalsState(l_quantum_symbol='s'),
                OrbitalsState(l_quantum_symbol='s'),
                (0, 0, 0),
                'sss',
            ),
            (
                OrbitalsState(l_quantum_symbol='s'),
                OrbitalsState(l_quantum_symbol='p'),
                (0, 0, 0),
                'sps',
            ),
        ],
    )
    def test_normalize(
        self,
        orbital_1: Optional[OrbitalsState],
        orbital_2: Optional[OrbitalsState],
        bravais_vector: Optional[tuple],
        result: Optional[str],
    ):
        """
        Test the `normalize` method.

        Args:
            orbital_1 (Optional[OrbitalsState]): The first `OrbitalsState` section.
            orbital_2 (Optional[OrbitalsState]): The second `OrbitalsState` section.
            bravais_vector (Optional[tuple]): The bravais vector.
            result (Optional[str]): The expected SK bond `name` after normalization.
        """
        sk_bond = SlaterKosterBond()
        atoms_state = AtomsState()
        _ = Simulation(
            model_system=[ModelSystem(cell=[AtomicCell(atoms_state=[atoms_state])])]
        )
        if orbital_1 is not None:
            atoms_state.orbitals_state.append(orbital_1)
            sk_bond.orbital_1 = atoms_state.orbitals_state[0]
        if orbital_2 is not None:
            atoms_state.orbitals_state.append(orbital_2)
            sk_bond.orbital_2 = atoms_state.orbitals_state[-1]
        if bravais_vector is not None and len(bravais_vector) != 0:
            sk_bond.bravais_vector = bravais_vector
        sk_bond.normalize(EntryArchive(), logger)
        assert sk_bond.name == result
