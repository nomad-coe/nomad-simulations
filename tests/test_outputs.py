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

from nomad_simulations.schema_packages.model_method import ModelMethod
from nomad_simulations.schema_packages.model_system import ModelSystem
from nomad_simulations.schema_packages.numerical_settings import SelfConsistency
from nomad_simulations.schema_packages.outputs import Outputs, SCFOutputs
from nomad_simulations.schema_packages.properties import ElectronicBandGap

from . import logger
from .conftest import generate_scf_electronic_band_gap_template, generate_simulation


class TestOutputs:
    """
    Test the `Outputs` class defined in `outputs.py`.
    """

    def test_number_of_properties(self):
        """
        Test how many properties are defined under `Outputs` and its order. This test is done in order to control better
        which properties are already defined and in which order to control their normalizations
        """
        outputs = Outputs()
        assert len(outputs.m_def.all_sub_sections) == 22
        defined_properties = [
            'fermi_levels',
            'chemical_potentials',
            'crystal_field_splittings',
            'hopping_matrices',
            'electronic_eigenvalues',
            'electronic_band_gaps',
            'electronic_dos',
            'fermi_surfaces',
            'electronic_band_structures',
            'occupancies',
            'electronic_greens_functions',
            'electronic_self_energies',
            'hybridization_functions',
            'quasiparticle_weights',
            'permittivities',
            'absorption_spectra',
            'xas_spectra',
            'total_energies',
            'kinetic_energies',
            'potential_energies',
            'total_forces',
            'temperatures',
        ]
        assert list(outputs.m_def.all_sub_sections.keys()) == defined_properties

    @pytest.mark.parametrize(
        'band_gaps, values, result_length, result',
        [
            # no properties to extract
            ([], [], 0, []),
            # non-spin polarized case
            ([ElectronicBandGap(variables=[])], [2.0], 0, []),
            # spin polarized case
            (
                [
                    ElectronicBandGap(variables=[], spin_channel=0),
                    ElectronicBandGap(variables=[], spin_channel=1),
                ],
                [1.0, 1.5],
                2,
                [
                    ElectronicBandGap(variables=[], spin_channel=0),
                    ElectronicBandGap(variables=[], spin_channel=1),
                ],
            ),
        ],
    )
    def test_extract_spin_polarized_properties(
        self,
        band_gaps: list[ElectronicBandGap],
        values: list[float],
        result_length: int,
        result: list[ElectronicBandGap],
    ):
        """
        Test the `extract_spin_polarized_property` method.

        Args:
            band_gaps (list[ElectronicBandGap]): The `ElectronicBandGap` sections to be stored under `Outputs`.
            values (list[float]): The values to be assigned to the `ElectronicBandGap` sections.
            result_length (int): The expected length extracted from `extract_spin_polarized_property`.
            result (list[ElectronicBandGap]): The expected result of the `extract_spin_polarized_property` method.
        """
        outputs = Outputs()

        for i, band_gap in enumerate(band_gaps):
            band_gap.value = values[i]
            outputs.electronic_band_gaps.append(band_gap)
        gaps = outputs.extract_spin_polarized_property(
            property_name='electronic_band_gaps'
        )
        assert len(gaps) == result_length
        if len(result) > 0:
            for i, result_gap in enumerate(result):
                result_gap.value = values[i]
                # ? comparing the sections does not work
                assert gaps[i].value == result_gap.value
        else:
            assert gaps == result

    @pytest.mark.parametrize(
        'model_system',
        [(None), (ModelSystem(name='example'))],
    )
    def test_set_model_system_ref(self, model_system: Optional[ModelSystem]):
        """
        Test the `set_model_system_ref` method.

        Args:
            model_system (Optional[ModelSystem]): The `ModelSystem` to be tested for the `model_system_ref` reference
            stored in `Outputs`.
        """
        outputs = Outputs()
        simulation = generate_simulation(model_system=model_system, outputs=outputs)
        model_system_ref = outputs.set_model_system_ref()
        if model_system is not None:
            assert model_system_ref == simulation.model_system[-1]
            assert model_system_ref.name == 'example'
        else:
            assert model_system_ref is None

    @pytest.mark.parametrize(
        'model_method',
        [(None), (ModelMethod(name='example'))],
    )
    def test_set_model_method_ref(self, model_method: Optional[ModelMethod]):
        """
        Test the `set_model_method_ref` method.

        Args:
            model_method (Optional[ModelMethod]): The `ModelMethod` to be tested for the `model_method_ref` reference
            stored in `Outputs`.
        """
        outputs = Outputs()
        simulation = generate_simulation(model_method=model_method, outputs=outputs)
        model_method_ref = outputs.set_model_method_ref()
        if model_method is not None:
            assert model_method_ref == simulation.model_method[-1]
            assert model_method_ref.name == 'example'
        else:
            assert model_method_ref is None

    @pytest.mark.parametrize(
        'model_system, model_method',
        [
            (None, None),
            (ModelSystem(name='example system'), None),
            (None, ModelMethod(name='example method')),
            (ModelSystem(name='example system'), ModelMethod(name='example method')),
        ],
    )
    def test_normalize(
        self, model_system: Optional[ModelSystem], model_method: Optional[ModelMethod]
    ):
        """
        Test the `normalize` method.

        Args:
            model_system (Optional[ModelSystem]): The expected `model_system_ref` obtained after normalization and
            initially stored under `Simulation.model_system[0]`.
            model_method (Optional[ModelMethod]): The expected `model_method_ref` obtained after normalization and
            initially stored under `Simulation.model_method[0]`.
        """
        outputs = Outputs()
        simulation = generate_simulation(
            model_system=model_system, model_method=model_method, outputs=outputs
        )
        outputs.normalize(archive=EntryArchive(), logger=logger)
        if model_system is not None:
            assert outputs.model_system_ref == simulation.model_system[-1]
            assert outputs.model_system_ref.name == 'example system'
        else:
            assert outputs.model_system_ref is None
        if model_method is not None:
            assert outputs.model_method_ref == simulation.model_method[-1]
            assert outputs.model_method_ref.name == 'example method'
        else:
            assert outputs.model_method_ref is None


class TestSCFOutputs:
    """
    Test the `SCFOutputs` class defined in `outputs.py`.
    """

    @pytest.mark.parametrize(
        'scf_last_steps, i_property, values, scf_parameters, result',
        [
            # empty `scf_last_steps`
            ([], 0, None, None, []),
            # length of `scf_last_steps` is different from 2
            ([Outputs()], 0, None, None, []),
            # no property matching `'electronic_band_gaps'` stored under the `scf_last_steps`
            ([Outputs(), Outputs()], 0, None, None, []),
            # `i_property` is out of range
            (
                [
                    Outputs(electronic_band_gaps=[ElectronicBandGap()]),
                    Outputs(electronic_band_gaps=[ElectronicBandGap()]),
                ],
                2,
                None,
                None,
                [],
            ),
            # no `value` stored in the `scf_last_steps` property
            (
                [
                    Outputs(electronic_band_gaps=[ElectronicBandGap()]),
                    Outputs(electronic_band_gaps=[ElectronicBandGap()]),
                ],
                0,
                None,
                None,
                [],
            ),
            # no `SelfConsistency` section and `threshold_change_unit` defined and macthing units for property `value` (`'joule'`)
            (
                [
                    Outputs(electronic_band_gaps=[ElectronicBandGap()]),
                    Outputs(electronic_band_gaps=[ElectronicBandGap()]),
                ],
                0,
                [1.0, 2.0],
                None,
                [],
            ),
            # valid case
            (
                [
                    Outputs(electronic_band_gaps=[ElectronicBandGap()]),
                    Outputs(electronic_band_gaps=[ElectronicBandGap()]),
                ],
                0,
                [1.0, 2.0],
                SelfConsistency(threshold_change=1e-6, threshold_change_unit='joule'),
                [1.0, 2.0],
            ),
        ],
    )
    def test_get_last_scf_steps_value(
        self,
        scf_last_steps: list[Outputs],
        i_property: int,
        values: list[float],
        scf_parameters: Optional[SelfConsistency],
        result: list[float],
    ):
        scf_outputs = SCFOutputs()
        for i, scf_step in enumerate(scf_last_steps):
            property_section = getattr(scf_step, 'electronic_band_gaps')
            if property_section is not None and values is not None:
                property_section[i_property].value = values[i]
        scf_values = scf_outputs.get_last_scf_steps_value(
            scf_last_steps=scf_last_steps,
            property_name='electronic_band_gaps',
            i_property=i_property,
            scf_parameters=scf_parameters,
            logger=logger,
        )
        assert scf_values == result

    @pytest.mark.parametrize(
        'n_scf_steps, threshold_change, property_name, i_property, result',
        [
            # `n_scf_steps` is less than 2
            (0, None, '', 0, False),
            # no `self_consistency_ref` section
            (5, None, '', 0, False),
            # no property matching `property_name`
            (5, None, 'fermi_levels', 0, False),
            # `i_property` is out of range
            (5, None, 'electronic_band_gaps', 2, False),
            # property is not converged
            (5, 1e-5, 'electronic_band_gaps', 0, False),
            # valid case: property is converged
            (5, 1e-3, 'electronic_band_gaps', 0, True),
        ],
    )
    def test_resolve_is_scf_converged(
        self,
        n_scf_steps: int,
        threshold_change: Optional[float],
        property_name: str,
        i_property: int,
        result: bool,
    ):
        """
        Test the `resolve_is_scf_converged` method.

        Args:
            n_scf_steps (int): The number of SCF steps to add under `SCFOutputs`.
            threshold_change (Optional[float]): The threshold change to be used for the SCF convergence.
            property_name (str): The name of the property to be tested for convergence.
            i_property (int): The index of the property to be tested for convergence.
            result (bool): The expected result of the `resolve_is_scf_converged` method.
        """
        scf_outputs = generate_scf_electronic_band_gap_template(
            n_scf_steps=n_scf_steps, threshold_change=threshold_change
        )
        is_scf_converged = scf_outputs.resolve_is_scf_converged(
            property_name=property_name,
            i_property=i_property,
            physical_property=scf_outputs.electronic_band_gaps[0],
            logger=logger,
        )
        assert is_scf_converged == result

    @pytest.mark.parametrize(
        'n_scf_steps, threshold_change, result',
        [
            # `n_scf_steps` is less than 2
            (0, None, False),
            # no `self_consistency_ref` section
            (5, None, False),
            # property is not converged
            (5, 1e-5, False),
            # valid case: property is converged
            (5, 1e-3, True),
        ],
    )
    def test_normalize(
        self,
        n_scf_steps: int,
        threshold_change: float,
        result: bool,
    ):
        """
        Test the `normalize` method.

        Args:
            n_scf_steps (int): The number of SCF steps to add under `SCFOutputs`.
            threshold_change (Optional[float]): The threshold change to be used for the SCF convergence.
            result (bool): The expected result after normalization and population of the `is_scf_converged` quantity.
        """
        scf_outputs = generate_scf_electronic_band_gap_template(
            n_scf_steps=n_scf_steps, threshold_change=threshold_change
        )
        scf_outputs.normalize(EntryArchive(), logger)
        assert scf_outputs.electronic_band_gaps[0].is_scf_converged == result
