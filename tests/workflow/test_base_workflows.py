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
from nomad.datamodel.metainfo.workflow import Link, Task, TaskReference

from nomad_simulations.schema_packages.model_method import (
    DFT,
    TB,
    ModelMethod,
)
from nomad_simulations.schema_packages.model_system import ModelSystem
from nomad_simulations.schema_packages.outputs import Outputs, SCFOutputs
from nomad_simulations.schema_packages.workflow import BeyondDFT, SinglePoint


class TestBeyondDFT:
    @pytest.mark.parametrize(
        'tasks, result',
        [
            # no task
            (None, None),
            # empty task
            ([Task()], []),
            # no outputs
            ([Task(name='task')], []),
            # one task with one output
            ([Task(outputs=[Link(section=Outputs())])], [Outputs]),
            # one task with two outputs (last one is SCF type)
            (
                [Task(outputs=[Link(section=Outputs()), Link(section=SCFOutputs())])],
                [SCFOutputs],
            ),
            # two tasks with one output each
            (
                [
                    Task(outputs=[Link(section=Outputs())]),
                    Task(outputs=[Link(section=SCFOutputs())]),
                ],
                [Outputs, SCFOutputs],
            ),
            # two tasks with two outputs each (note order of the last outputs types)
            (
                [
                    Task(outputs=[Link(section=Outputs()), Link(section=SCFOutputs())]),
                    Task(outputs=[Link(section=SCFOutputs()), Link(section=Outputs())]),
                ],
                [SCFOutputs, Outputs],
            ),
        ],
    )
    def test_resolve_all_outputs(self, tasks: list[Task], result: list[Outputs]):
        """
        Test the `resolve_all_outputs` method of the `BeyondDFT` section.
        """
        workflow = BeyondDFT()
        workflow.tasks = tasks
        all_outputs = workflow.resolve_all_outputs()
        if not result:
            assert all_outputs == result
        else:
            # ! comparing directly does not work becasue one is a section, the other a reference
            for i, output in enumerate(all_outputs):
                assert isinstance(output.section, result[i])

    @pytest.mark.parametrize(
        'tasks, result',
        [
            # no task
            (None, None),
            ([TaskReference()], []),
            ([TaskReference(), TaskReference()], []),
            (
                [TaskReference(task=SinglePoint()), TaskReference(task=SinglePoint())],
                [],
            ),
            (
                [
                    TaskReference(
                        task=SinglePoint(inputs=[Link(section=ModelSystem())])
                    ),
                    TaskReference(
                        task=SinglePoint(inputs=[Link(section=ModelSystem())])
                    ),
                ],
                [],
            ),
            (
                [
                    TaskReference(
                        task=SinglePoint(
                            inputs=[
                                Link(section=ModelSystem()),
                                Link(section=DFT()),
                            ]
                        )
                    ),
                    TaskReference(
                        task=SinglePoint(
                            inputs=[
                                Link(section=ModelSystem()),
                            ]
                        )
                    ),
                ],
                [DFT],
            ),
            (
                [
                    TaskReference(
                        task=SinglePoint(
                            inputs=[
                                Link(section=ModelSystem()),
                                Link(section=DFT()),
                            ]
                        )
                    ),
                    TaskReference(
                        task=SinglePoint(
                            inputs=[
                                Link(section=ModelSystem()),
                                Link(section=TB()),
                            ]
                        )
                    ),
                ],
                [DFT, TB],
            ),
        ],
    )
    def test_resolve_method_refs(
        self, tasks: list[TaskReference], result: list[ModelMethod]
    ):
        """
        Test the `resolve_method_refs` method of the `BeyondDFT` section.
        """
        workflow = BeyondDFT()
        workflow.tasks = tasks
        method_refs = workflow.resolve_method_refs(
            tasks=workflow.tasks,
            tasks_names=['DFT SinglePoint Task', 'TB SinglePoint Task'],
        )

        if tasks is not None and len(tasks) == 2:
            assert workflow.tasks[0].name == 'DFT SinglePoint Task'
            assert workflow.tasks[1].name == 'TB SinglePoint Task'
        if not result:
            assert method_refs == result
        else:
            # ! comparing directly does not work becasue one is a section, the other a reference
            for i, method in enumerate(result):
                assert isinstance(method_refs[i], method)
