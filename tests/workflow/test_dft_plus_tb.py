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
from nomad.datamodel.metainfo.workflow import Link, TaskReference

from nomad_simulations.schema_packages.model_method import (
    DFT,
    TB,
    BaseModelMethod,
    ModelMethod,
)
from nomad_simulations.schema_packages.model_system import ModelSystem
from nomad_simulations.schema_packages.outputs import Outputs, SCFOutputs
from nomad_simulations.schema_packages.workflow import DFTPlusTB, SinglePoint

from ..conftest import generate_simulation
from . import logger


class TestDFTPlusTB:
    @pytest.mark.parametrize(
        'inputs, outputs, tasks, result_tasks',
        [
            # no inputs, outputs, tasks
            (None, None, None, []),
            # only 1 task
            (None, None, [TaskReference()], []),
            # empty tasks
            (
                None,
                None,
                [TaskReference(), TaskReference()],
                [],
            ),
            # only one task is populated
            (
                None,
                None,
                [
                    TaskReference(task=SinglePoint()),
                    TaskReference(),
                ],
                [],
            ),
            # only one task is populated with inputs
            (
                None,
                None,
                [
                    TaskReference(task=SinglePoint(inputs=[Link()])),
                    TaskReference(task=SinglePoint()),
                ],
                [],
            ),
            # only one task is populated with outputs
            (
                None,
                None,
                [
                    TaskReference(task=SinglePoint(outputs=[Link(name='output dft')])),
                    TaskReference(task=SinglePoint()),
                ],
                [],
            ),
            # positive testing
            (
                [Link(name='input system')],
                [Link(name='output tb')],
                [
                    TaskReference(task=SinglePoint(outputs=[Link(name='output dft')])),
                    TaskReference(task=SinglePoint()),
                ],
                [
                    TaskReference(
                        task=SinglePoint(outputs=[Link(name='output dft')]),
                        inputs=[Link(name='Input Model System')],
                        outputs=[Link(name='Output DFT Data')],
                    ),
                    TaskReference(
                        task=SinglePoint(),
                        inputs=[Link(name='Output DFT Data')],
                        outputs=[Link(name='Output TB Data')],
                    ),
                ],
            ),
        ],
    )
    def test_link_task_inputs_outputs(
        self,
        inputs: list[Link],
        outputs: list[Link],
        tasks: list[TaskReference],
        result_tasks: list[TaskReference],
    ):
        """
        Test the `link_task_inputs_outputs` method of the `DFTPlusTB` section.
        """
        workflow = DFTPlusTB()
        workflow.tasks = tasks
        workflow.inputs = inputs
        workflow.outputs = outputs

        workflow.link_task_inputs_outputs(tasks=workflow.tasks, logger=logger)

        if not result_tasks:
            assert not workflow.m_xpath('tasks[0].inputs') and not workflow.m_xpath(
                'tasks[0].outputs'
            )
            assert not workflow.m_xpath('tasks[1].inputs') and not workflow.m_xpath(
                'tasks[1].outputs'
            )
        else:
            for i, task in enumerate(workflow.tasks):
                assert task.inputs[0].name == result_tasks[i].inputs[0].name
                assert task.outputs[0].name == result_tasks[i].outputs[0].name

    @pytest.mark.parametrize(
        'inputs, outputs, tasks, result_name, result_methods, result_tasks',
        [
            # all none
            (None, None, None, None, None, []),
            # only one task
            (None, None, [TaskReference()], None, None, []),
            # two empty tasks
            (None, None, [TaskReference(), TaskReference()], None, None, []),
            # only one task has a task
            (
                None,
                None,
                [TaskReference(task=SinglePoint()), TaskReference()],
                None,
                None,
                [],
            ),
            # both tasks with empty task sections, one is not SinglePoint
            (
                None,
                None,
                [TaskReference(task=DFTPlusTB()), TaskReference(task=SinglePoint())],
                None,
                None,
                [],
            ),
            # both tasks with empty SinglePoint task sections; name is resolved
            (
                None,
                None,
                [TaskReference(task=SinglePoint()), TaskReference(task=SinglePoint())],
                'DFT+TB',
                None,
                [],
            ),
            # both tasks have input for ModelSystem
            (
                None,
                None,
                [
                    TaskReference(
                        task=SinglePoint(
                            inputs=[Link(name='input system', section=ModelSystem())]
                        )
                    ),
                    TaskReference(
                        task=SinglePoint(
                            inputs=[Link(name='input system', section=ModelSystem())]
                        )
                    ),
                ],
                'DFT+TB',
                None,
                [],
            ),
            # one task has an input with a ref to DFT section
            (
                None,
                None,
                [
                    TaskReference(
                        task=SinglePoint(
                            inputs=[
                                Link(name='input system', section=ModelSystem()),
                                Link(name='dft method', section=DFT()),
                            ]
                        )
                    ),
                    TaskReference(
                        task=SinglePoint(
                            inputs=[Link(name='input system', section=ModelSystem())]
                        )
                    ),
                ],
                'DFT+TB',
                [DFT],
                [],
            ),
            # both tasks have inputs with refs to DFT and TB sections
            (
                None,
                None,
                [
                    TaskReference(
                        task=SinglePoint(
                            inputs=[
                                Link(name='input system', section=ModelSystem()),
                                Link(name='dft method', section=DFT()),
                            ]
                        )
                    ),
                    TaskReference(
                        task=SinglePoint(
                            inputs=[
                                Link(name='input system', section=ModelSystem()),
                                Link(name='tb method', section=TB()),
                            ]
                        )
                    ),
                ],
                'DFT+TB',
                [DFT, TB],
                [],
            ),
            # one task has an output, but the workflow inputs and outputs are empty
            (
                None,
                None,
                [
                    TaskReference(
                        task=SinglePoint(
                            inputs=[
                                Link(name='input system', section=ModelSystem()),
                                Link(name='dft method', section=DFT()),
                            ],
                            outputs=[Link(name='output dft', section=Outputs())],
                        )
                    ),
                    TaskReference(
                        task=SinglePoint(
                            inputs=[
                                Link(name='input system', section=ModelSystem()),
                                Link(name='tb method', section=TB()),
                            ],
                        )
                    ),
                ],
                'DFT+TB',
                [DFT, TB],
                [],
            ),
            # positive testing
            (
                [Link(name='input system')],
                [Link(name='output tb')],
                [
                    TaskReference(
                        task=SinglePoint(
                            inputs=[
                                Link(name='input system', section=ModelSystem()),
                                Link(name='dft method', section=DFT()),
                            ],
                            outputs=[Link(name='output dft', section=Outputs())],
                        )
                    ),
                    TaskReference(
                        task=SinglePoint(
                            inputs=[
                                Link(name='input system', section=ModelSystem()),
                                Link(name='tb method', section=TB()),
                            ],
                            outputs=[Link(name='output tb', section=Outputs())],
                        )
                    ),
                ],
                'DFT+TB',
                [DFT, TB],
                [
                    TaskReference(
                        task=SinglePoint(outputs=[Link(name='output dft')]),
                        inputs=[Link(name='Input Model System')],
                        outputs=[Link(name='Output DFT Data')],
                    ),
                    TaskReference(
                        task=SinglePoint(),
                        inputs=[Link(name='Output DFT Data')],
                        outputs=[Link(name='Output TB Data')],
                    ),
                ],
            ),
        ],
    )
    def test_normalize(
        self,
        inputs: list[Link],
        outputs: list[Link],
        tasks: list[TaskReference],
        result_name: Optional[str],
        result_methods: Optional[list[ModelMethod]],
        result_tasks: Optional[list[TaskReference]],
    ):
        """
        Test the `normalize` method of the `DFTPlusTB` section.
        """
        archive = EntryArchive()

        # Add `Simulation` to archive
        simulation = generate_simulation(
            model_system=ModelSystem(), model_method=ModelMethod(), outputs=Outputs()
        )
        archive.data = simulation

        # Add `SinglePoint` to archive
        workflow = DFTPlusTB()
        workflow.inputs = inputs
        workflow.outputs = outputs
        workflow.tasks = tasks
        archive.workflow2 = workflow

        workflow.normalize(archive=archive, logger=logger)

        # Test `name` of the workflow
        assert workflow.name == result_name

        # Test `method` of the workflow
        if len(result_tasks) > 0:
            assert workflow.tasks[0].name == 'DFT SinglePoint Task'
            assert workflow.tasks[1].name == 'TB SinglePoint Task'
        if not result_methods:
            assert not workflow.m_xpath(
                'method.dft_method_ref'
            ) and not workflow.m_xpath('method.tb_method_ref')
        else:
            # ! comparing directly does not work becasue one is a section, the other a reference
            assert isinstance(workflow.method.dft_method_ref, result_methods[0])
            if len(result_methods) == 2:
                assert isinstance(workflow.method.tb_method_ref, result_methods[1])

        # Test `tasks` of the workflow
        if not result_tasks:
            assert not workflow.m_xpath('tasks[0].inputs') and not workflow.m_xpath(
                'tasks[0].outputs'
            )
            assert not workflow.m_xpath('tasks[1].inputs') and not workflow.m_xpath(
                'tasks[1].outputs'
            )
        else:
            for i, task in enumerate(workflow.tasks):
                assert task.inputs[0].name == result_tasks[i].inputs[0].name
                assert task.outputs[0].name == result_tasks[i].outputs[0].name
