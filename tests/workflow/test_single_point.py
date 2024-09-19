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

from nomad_simulations.schema_packages.model_method import ModelMethod
from nomad_simulations.schema_packages.model_system import ModelSystem
from nomad_simulations.schema_packages.outputs import Outputs, SCFOutputs
from nomad_simulations.schema_packages.workflow import SinglePoint

from ..conftest import generate_simulation
from . import logger


class TestSinglePoint:
    @pytest.mark.parametrize(
        'model_system, model_method, outputs, result',
        [
            # empty sections in archive.data
            (None, None, None, Task()),
            # only one section in archive.data
            (ModelSystem(), None, None, Task()),
            # another section in archive.data
            (None, ModelMethod(), None, Task()),
            # only two sections in archive.data
            (ModelSystem(), ModelMethod(), None, Task()),
            # all sections in archive.data
            (
                ModelSystem(),
                ModelMethod(),
                Outputs(),
                Task(
                    inputs=[
                        Link(name='Input Model System', section=ModelSystem()),
                        Link(name='Input Model Method', section=ModelMethod()),
                    ],
                    outputs=[
                        Link(name='Output Data', section=Outputs()),
                    ],
                ),
            ),
        ],
    )
    def test_generate_task(
        self,
        model_system: Optional[ModelSystem],
        model_method: Optional[ModelMethod],
        outputs: Optional[Outputs],
        result: Task,
    ):
        """
        Test the `generate_task` method of the `SinglePoint` section.
        """
        archive = EntryArchive()
        simulation = generate_simulation(
            model_system=model_system, model_method=model_method, outputs=outputs
        )
        archive.data = simulation
        workflow = SinglePoint()
        archive.workflow2 = workflow

        single_point_task = workflow.generate_task(archive=archive, logger=logger)
        if not result.inputs:
            assert isinstance(single_point_task, Task)
            assert not single_point_task.inputs and not single_point_task.outputs
        else:
            assert single_point_task.inputs[0].name == result.inputs[0].name
            assert single_point_task.inputs[1].name == result.inputs[1].name
            assert single_point_task.outputs[0].name == result.outputs[0].name

    @pytest.mark.parametrize(
        'scf_output, result',
        [
            # no outputs
            (None, 1),
            # output is not of type SCFOutputs
            (Outputs(), 1),
            # SCFOutputs without scf_steps
            (SCFOutputs(), 1),
            # 3 scf_steps
            (SCFOutputs(scf_steps=[Outputs(), Outputs(), Outputs()]), 3),
        ],
    )
    def test_resolve_n_scf_steps(self, scf_output: Outputs, result: int):
        """
        Test the `resolve_n_scf_steps` method of the `SinglePoint` section.
        """
        archive = EntryArchive()
        simulation = generate_simulation(
            model_system=ModelSystem(), model_method=ModelMethod(), outputs=scf_output
        )
        archive.data = simulation
        workflow = SinglePoint()
        archive.workflow2 = workflow

        # Add the scf output to the workflow.outputs
        if scf_output is not None:
            workflow.outputs = [
                Link(name='SCF Output Data', section=archive.data.outputs[-1])
            ]

        n_scf_steps = workflow.resolve_n_scf_steps()
        assert n_scf_steps == result

    @pytest.mark.parametrize(
        'model_system, model_method, outputs, tasks, result_task, result_n_scf_steps',
        [
            # multiple tasks being stored in SinglePoint
            (
                ModelSystem(),
                ModelMethod(),
                Outputs(),
                [Task(name='task 1'), Task(name='task 2')],
                [],
                None,
            ),
            # only one task is being stored in SinglePoint
            (
                ModelSystem(),
                ModelMethod(),
                Outputs(),
                [Task(name='parsed task')],
                [Task(name='parsed task')],
                1,
            ),
            # no archive sections (empty generated task)
            (None, None, None, None, [Task(name='generated task')], 1),
            # only one section in archive.data
            (ModelSystem(), None, None, None, [Task(name='generated task')], 1),
            # another section in archive.data
            (None, ModelMethod(), None, None, [Task(name='generated task')], 1),
            # only two sections in archive.data
            (
                ModelSystem(),
                ModelMethod(),
                None,
                None,
                [Task(name='generated task')],
                1,
            ),
            # all sections in archive.data, so generated task has inputs and outputs
            (
                ModelSystem(),
                ModelMethod(),
                Outputs(),
                None,
                [
                    Task(
                        name='generated task',
                        inputs=[
                            Link(name='Input Model System', section=ModelSystem()),
                            Link(name='Input Model Method', section=ModelMethod()),
                        ],
                        outputs=[
                            Link(name='Output Data', section=Outputs()),
                        ],
                    )
                ],
                1,
            ),
            # Outputs is SCFOutputs but no scf_steps
            (
                ModelSystem(),
                ModelMethod(),
                SCFOutputs(),
                None,
                [
                    Task(
                        name='generated task',
                        inputs=[
                            Link(name='Input Model System', section=ModelSystem()),
                            Link(name='Input Model Method', section=ModelMethod()),
                        ],
                        outputs=[
                            Link(name='Output Data', section=SCFOutputs()),
                        ],
                    )
                ],
                1,
            ),
            # 3 scf_steps
            (
                ModelSystem(),
                ModelMethod(),
                SCFOutputs(scf_steps=[Outputs(), Outputs(), Outputs()]),
                None,
                [
                    Task(
                        name='generated task',
                        inputs=[
                            Link(name='Input Model System', section=ModelSystem()),
                            Link(name='Input Model Method', section=ModelMethod()),
                        ],
                        outputs=[
                            Link(
                                name='Output Data',
                                section=SCFOutputs(
                                    scf_steps=[Outputs(), Outputs(), Outputs()]
                                ),
                            ),
                        ],
                    )
                ],
                3,
            ),
        ],
    )
    def test_normalize(
        self,
        model_system: Optional[ModelSystem],
        model_method: Optional[ModelMethod],
        outputs: Optional[Outputs],
        tasks: list[Task],
        result_task: list[Task],
        result_n_scf_steps: int,
    ):
        """
        Test the `normalize` method of the `SinglePoint` section.
        """
        archive = EntryArchive()
        simulation = generate_simulation(
            model_system=model_system, model_method=model_method, outputs=outputs
        )
        archive.data = simulation
        workflow = SinglePoint()
        archive.workflow2 = workflow

        if tasks is not None:
            workflow.tasks = tasks

        workflow.normalize(archive=archive, logger=logger)

        if not result_task:
            assert workflow.tasks == result_task
        else:
            single_point_task = workflow.tasks[0]
            if not result_task[0].inputs:
                assert isinstance(single_point_task, Task)
                assert not single_point_task.inputs and not single_point_task.outputs
            else:
                assert single_point_task.inputs[0].name == result_task[0].inputs[0].name
                assert single_point_task.inputs[1].name == result_task[0].inputs[1].name
                assert (
                    single_point_task.outputs[0].name == result_task[0].outputs[0].name
                )
        assert workflow.n_scf_steps == result_n_scf_steps
