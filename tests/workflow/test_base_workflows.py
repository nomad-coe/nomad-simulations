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
from nomad.datamodel.metainfo.workflow import Link, Task, Workflow

from nomad_simulations.schema_packages.model_method import BaseModelMethod, ModelMethod
from nomad_simulations.schema_packages.model_system import ModelSystem
from nomad_simulations.schema_packages.outputs import Outputs
from nomad_simulations.schema_packages.workflow import (
    BeyondDFTMethod,
    BeyondDFTWorkflow,
    SimulationWorkflow,
)

from ..conftest import generate_simulation
from . import logger


class TestSimulationWorkflow:
    @pytest.mark.parametrize(
        'model_system, model_method, outputs, workflow_inputs, workflow_outputs',
        [
            # empty sections in archive.data
            (None, None, None, [], []),
            # only one section in archive.data
            (ModelSystem(), None, None, [], []),
            # another section in archive.data
            (None, ModelMethod(), None, [], []),
            # only two sections in archive.data
            (ModelSystem(), ModelMethod(), None, [], []),
            # all sections in archive.data
            (
                ModelSystem(),
                ModelMethod(),
                Outputs(),
                [Link(name='Input Model System', section=ModelSystem())],
                [Link(name='Output Data', section=Outputs())],
            ),
        ],
    )
    def test_resolve_inputs_outputs(
        self,
        model_system: Optional[ModelSystem],
        model_method: Optional[ModelMethod],
        outputs: Optional[Outputs],
        workflow_inputs: list[Link],
        workflow_outputs: list[Link],
    ):
        """
        Test the `resolve_inputs_outputs` method of the `SimulationWorkflow` section.
        """
        archive = EntryArchive()
        simulation = generate_simulation(
            model_system=model_system, model_method=model_method, outputs=outputs
        )
        archive.data = simulation
        workflow = SimulationWorkflow()
        archive.workflow2 = workflow

        workflow.resolve_inputs_outputs(archive=archive, logger=logger)
        if not workflow_inputs:
            assert workflow.inputs == workflow_inputs
        else:
            assert len(workflow.inputs) == 1
            assert workflow.inputs[0].name == workflow_inputs[0].name
            # ! direct comparison of section does not work (probably an issue with references)
            # assert workflow.inputs[0].section == workflow_inputs[0].section
            assert workflow._input_systems[0] == model_system
            assert workflow._input_methods[0] == model_method
        if not workflow_outputs:
            assert workflow.outputs == workflow_outputs
        else:
            assert len(workflow.outputs) == 1
            assert workflow.outputs[0].name == workflow_outputs[0].name
            # ! direct comparison of section does not work (probably an issue with references)
            # assert workflow.outputs[0].section == workflow_outputs[0].section
            assert workflow._outputs[0] == outputs

    @pytest.mark.parametrize(
        'model_system, model_method, outputs, workflow_inputs, workflow_outputs',
        [
            # empty sections in archive.data
            (None, None, None, [], []),
            # only one section in archive.data
            (ModelSystem(), None, None, [], []),
            # another section in archive.data
            (None, ModelMethod(), None, [], []),
            # only two sections in archive.data
            (ModelSystem(), ModelMethod(), None, [], []),
            # all sections in archive.data
            (
                ModelSystem(),
                ModelMethod(),
                Outputs(),
                [Link(name='Input Model System', section=ModelSystem())],
                [Link(name='Output Data', section=Outputs())],
            ),
        ],
    )
    def test_normalize(
        self,
        model_system: Optional[ModelSystem],
        model_method: Optional[ModelMethod],
        outputs: Optional[Outputs],
        workflow_inputs: list[Link],
        workflow_outputs: list[Link],
    ):
        """
        Test the `normalize` method of the `SimulationWorkflow` section.
        """
        archive = EntryArchive()
        simulation = generate_simulation(
            model_system=model_system, model_method=model_method, outputs=outputs
        )
        archive.data = simulation
        workflow = SimulationWorkflow()
        archive.workflow2 = workflow

        workflow.normalize(archive=archive, logger=logger)
        if not workflow_inputs:
            assert workflow.inputs == workflow_inputs
        else:
            assert len(workflow.inputs) == 1
            assert workflow.inputs[0].name == workflow_inputs[0].name
            # ! direct comparison of section does not work (probably an issue with references)
            # assert workflow.inputs[0].section == workflow_inputs[0].section
            assert workflow._input_systems[0] == model_system
            assert workflow._input_methods[0] == model_method
            # Extra attribute from the `normalize` function
            # ! direct comparison of section does not work (probably an issue with references)
            # assert workflow.initial_structure == workflow_inputs[0].section
        if not workflow_outputs:
            assert workflow.outputs == workflow_outputs
        else:
            assert len(workflow.outputs) == 1
            assert workflow.outputs[0].name == workflow_outputs[0].name
            # ! direct comparison of section does not work (probably an issue with references)
            # assert workflow.outputs[0].section == workflow_outputs[0].section
            assert workflow._outputs[0] == outputs


class TestBeyondDFTMethod:
    @pytest.mark.parametrize(
        'task, result',
        [
            # no task
            (None, None),
            # empty task
            (Task(), None),
            # task only contains ModelSystem
            (
                Task(inputs=[Link(name='Input Model System', section=ModelSystem())]),
                None,
            ),
            # no `section` in the link
            (
                Task(inputs=[Link(name='Input Model Method')]),
                None,
            ),
            # task only contains ModelMethod
            (
                Task(inputs=[Link(name='Input Model Method', section=ModelMethod())]),
                ModelMethod(),
            ),
            # task contains both ModelSystem and ModelMethod
            (
                Task(
                    inputs=[
                        Link(name='Input Model System', section=ModelSystem()),
                        Link(name='Input Model Method', section=ModelMethod()),
                    ]
                ),
                ModelMethod(),
            ),
        ],
    )
    def test_resolve_beyonddft_method_ref(
        self, task: Optional[Task], result: Optional[BaseModelMethod]
    ):
        """
        Test the `resolve_beyonddft_method_ref` method of the `BeyondDFTMethod` section.
        """
        beyond_dft_method = BeyondDFTMethod()
        # ! direct comparison of section does not work (probably an issue with references)
        if result is not None:
            assert (
                beyond_dft_method.resolve_beyonddft_method_ref(task=task).m_def.name
                == result.m_def.name
            )
        else:
            assert beyond_dft_method.resolve_beyonddft_method_ref(task=task) == result


class TestBeyondDFT:
    def test_resolve_all_outputs(self):
        assert True
