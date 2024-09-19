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
from nomad.datamodel.metainfo.workflow import Link, Task, TaskReference, Workflow

from nomad_simulations.schema_packages.model_method import (
    DFT,
    TB,
    BaseModelMethod,
    ModelMethod,
)
from nomad_simulations.schema_packages.model_system import ModelSystem
from nomad_simulations.schema_packages.outputs import Outputs
from nomad_simulations.schema_packages.workflow import (
    DFTPlusTB,
    DFTPlusTBMethod,
)

from ..conftest import generate_simulation
from . import logger


class TestDFTPlusTB:
    @pytest.mark.parametrize(
        'tasks, result',
        [
            (None, None),
            ([TaskReference(name='dft')], None),
            (
                [
                    TaskReference(name='dft'),
                    TaskReference(name='tb 1'),
                    TaskReference(name='tb 2'),
                ],
                None,
            ),
            ([TaskReference(name='dft'), TaskReference(name='tb')], None),
            (
                [
                    TaskReference(name='dft', task=Task(name='dft task')),
                    TaskReference(name='tb'),
                ],
                None,
            ),
            (
                [
                    TaskReference(
                        name='dft',
                        task=Task(
                            name='dft task',
                            inputs=[
                                Link(name='model system', section=ModelSystem()),
                                Link(name='model method dft', section=DFT()),
                            ],
                        ),
                    ),
                    TaskReference(
                        name='tb',
                        task=Task(name='tb task'),
                    ),
                ],
                [DFT, None],
            ),
            (
                [
                    TaskReference(
                        name='dft',
                        task=Task(
                            name='dft task',
                            inputs=[
                                Link(name='model system', section=ModelSystem()),
                                Link(name='model method dft', section=DFT()),
                            ],
                        ),
                    ),
                    TaskReference(
                        name='tb',
                        task=Task(
                            name='tb task',
                            inputs=[
                                Link(name='model system', section=ModelSystem()),
                                Link(name='model method tb', section=TB()),
                            ],
                        ),
                    ),
                ],
                [DFT, TB],
            ),
        ],
    )
    def test_resolve_method(
        self,
        tasks: list[Task],
        result: DFTPlusTBMethod,
    ):
        """
        Test the `resolve_method` method of the `DFTPlusTB` section.
        """
        archive = EntryArchive()
        workflow = DFTPlusTB()
        archive.workflow2 = workflow
        workflow.tasks = tasks
        workflow_method = workflow.resolve_method()
        if workflow_method is None:
            assert workflow_method == result
        else:
            if result[0] is not None:
                assert isinstance(workflow_method.dft_method_ref, result[0])
            else:
                assert workflow_method.dft_method_ref == result[0]
            if result[1] is not None:
                assert isinstance(workflow_method.tb_method_ref, result[1])
            else:
                assert workflow_method.tb_method_ref == result[1]

    def test_link_tasks(self):
        """
        Test the `resolve_n_scf_steps` method of the `DFTPlusTB` section.
        """
        archive = EntryArchive()
        workflow = DFTPlusTB()
        archive.workflow2 = workflow
        workflow.tasks = [
            TaskReference(
                name='dft',
                task=Task(
                    name='dft task',
                    inputs=[
                        Link(name='model system', section=ModelSystem()),
                        Link(name='model method dft', section=DFT()),
                    ],
                    outputs=[
                        Link(name='output dft', section=Outputs()),
                    ],
                ),
            ),
            TaskReference(
                name='tb',
                task=Task(
                    name='tb task',
                    inputs=[
                        Link(name='model system', section=ModelSystem()),
                        Link(name='model method tb', section=TB()),
                    ],
                    outputs=[
                        Link(name='output tb', section=Outputs()),
                    ],
                ),
            ),
        ]
        workflow.inputs = [Link(name='model system', section=ModelSystem())]
        workflow.outputs = [Link(name='output tb', section=Outputs())]

        # Linking and overwritting inputs and outputs
        workflow.link_tasks()

        dft_task = workflow.tasks[0]
        assert len(dft_task.inputs) == 1
        assert dft_task.inputs[0].name == 'Input Model System'
        assert len(dft_task.outputs) == 1
        assert dft_task.outputs[0].name == 'Output DFT Data'
        tb_task = workflow.tasks[1]
        assert len(tb_task.inputs) == 1
        assert tb_task.inputs[0].name == 'Output DFT Data'
        assert len(tb_task.outputs) == 1
        assert tb_task.outputs[0].name == 'Output TB Data'

    def test_overwrite_fermi_level(self):
        """
        Test the `overwrite_fermi_level` method of the `DFTPlusTB` section.
        """
        assert True

    def test_normalize(self):
        """
        Test the `normalize` method of the `DFTPlusTB` section.
        """
        assert True
