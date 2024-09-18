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
from nomad.datamodel.metainfo.workflow import Link, Task

from nomad_simulations.schema_packages.model_method import BaseModelMethod, ModelMethod
from nomad_simulations.schema_packages.model_system import ModelSystem
from nomad_simulations.schema_packages.outputs import Outputs
from nomad_simulations.schema_packages.workflow import (
    BeyondDFT,
    BeyondDFTMethod,
    SimulationWorkflow,
)

from ..conftest import generate_simulation
from . import logger


class TestSimulationWorkflow:
    @pytest.mark.parametrize(
        'model_system, model_method, outputs',
        [
            # empty sections in archive.data
            (None, None, None),
            # only one section in archive.data
            (ModelSystem(), None, None),
            # another section in archive.data
            (None, ModelMethod(), None),
            # only two sections in archive.data
            (ModelSystem(), ModelMethod(), None),
            # all sections in archive.data
            (ModelSystem(), ModelMethod(), Outputs()),
        ],
    )
    def test_resolve_inputs_outputs_from_archive(
        self,
        model_system: Optional[ModelSystem],
        model_method: Optional[ModelMethod],
        outputs: Optional[Outputs],
    ):
        """
        Test the `_resolve_inputs_outputs_from_archive` method of the `SimulationWorkflow` section.
        """
        archive = EntryArchive()
        simulation = generate_simulation(
            model_system=model_system, model_method=model_method, outputs=outputs
        )
        archive.data = simulation
        workflow = SimulationWorkflow()
        archive.workflow2 = workflow
        workflow._resolve_inputs_outputs_from_archive(archive=archive, logger=logger)
        if (
            model_system is not None
            and model_method is not None
            and outputs is not None
        ):
            for input_system in workflow._input_systems:
                assert isinstance(input_system, ModelSystem)
            for input_method in workflow._input_methods:
                assert isinstance(input_method, ModelMethod)
            for output in workflow._outputs:
                assert isinstance(output, Outputs)
        else:
            assert not workflow._input_systems
            assert not workflow._input_methods
            assert not workflow._outputs

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
        if not workflow_outputs:
            assert workflow.outputs == workflow_outputs
        else:
            assert len(workflow.outputs) == 1
            assert workflow.outputs[0].name == workflow_outputs[0].name
            # ! direct comparison of section does not work (probably an issue with references)
            # assert workflow.outputs[0].section == workflow_outputs[0].section

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
    @pytest.mark.parametrize(
        'tasks, result',
        [
            # no task
            (None, []),
            # empty task
            ([Task()], []),
            # task only contains inputs
            (
                [Task(inputs=[Link(name='Input Model System', section=ModelSystem())])],
                [],
            ),
            # one task with one output
            (
                [Task(outputs=[Link(name='Output Data 1', section=Outputs())])],
                [Link(name='Output Data 1', section=Outputs())],
            ),
            # one task with multiple outputs (only last is resolved)
            (
                [
                    Task(
                        outputs=[
                            Link(name='Output Data 1', section=Outputs()),
                            Link(name='Output Data 2', section=Outputs()),
                        ]
                    )
                ],
                [Link(name='Output Data 2', section=Outputs())],
            ),
            # multiple task with one output each
            (
                [
                    Task(
                        outputs=[Link(name='Task 1:Output Data 1', section=Outputs())]
                    ),
                    Task(
                        outputs=[Link(name='Task 2:Output Data 1', section=Outputs())]
                    ),
                ],
                [
                    Link(name='Task 1:Output Data 1', section=Outputs()),
                    Link(name='Task 2:Output Data 1', section=Outputs()),
                ],
            ),
            # multiple task with two outputs each (only last is resolved)
            (
                [
                    Task(
                        outputs=[
                            Link(name='Task 1:Output Data 1', section=Outputs()),
                            Link(name='Task 1:Output Data 2', section=Outputs()),
                        ]
                    ),
                    Task(
                        outputs=[
                            Link(name='Task 2:Output Data 1', section=Outputs()),
                            Link(name='Task 2:Output Data 2', section=Outputs()),
                        ]
                    ),
                ],
                [
                    Link(name='Task 1:Output Data 2', section=Outputs()),
                    Link(name='Task 2:Output Data 2', section=Outputs()),
                ],
            ),
        ],
    )
    def test_resolve_all_outputs(
        self, tasks: Optional[list[Task]], result: list[Outputs]
    ):
        """
        Test the `resolve_all_outputs` method of the `BeyondDFT` section.
        """
        workflow = BeyondDFT()
        if tasks is not None:
            workflow.tasks = tasks
        if result is not None:
            for i, output in enumerate(workflow.resolve_all_outputs()):
                assert output.name == result[i].name
        else:
            assert workflow.resolve_all_outputs() == result
