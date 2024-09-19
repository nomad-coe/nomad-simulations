#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD.
# See https://nomad-lab.eu for further info.
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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger

    from nomad_simulations.schema_packages.workflow import SinglePoint

from nomad.datamodel.metainfo.workflow import Link, TaskReference
from nomad.metainfo import Quantity, Reference

from nomad_simulations.schema_packages.model_method import DFT, TB, ModelMethod
from nomad_simulations.schema_packages.workflow import (
    BeyondDFT,
    BeyondDFTMethod,
)
from nomad_simulations.schema_packages.workflow.base_workflows import check_n_tasks


class DFTPlusTBMethod(BeyondDFTMethod):
    """
    Section used to reference the `DFT` and `TB` `ModelMethod` sections in each of the archives
    conforming a DFT+TB simulation workflow.
    """

    dft_method_ref = Quantity(
        type=Reference(DFT),
        description="""
        Reference to the DFT `ModelMethod` section in the DFT task.
        """,
    )
    tb_method_ref = Quantity(
        type=Reference(TB),
        description="""
        Reference to the TB `ModelMethod` section in the TB task.
        """,
    )


class DFTPlusTB(BeyondDFT):
    """
    A base section used to represent a DFT+TB calculation workflow. The `DFTPlusTB` workflow is composed of
    two tasks: the initial DFT calculation + the final TB projection.

    The section only needs to be populated with (everything else is handled by the `normalize` function):
        i. The `tasks` as `TaskReference` sections, adding `task` to the specific archive.workflow2 sections.
        ii. The `inputs` and `outputs` as `Link` sections pointing to the specific archives.

    Note 1: the `inputs[0]` of the `DFTPlusTB` coincides with the `inputs[0]` of the DFT task (`ModelSystem` section).
    Note 2: the `outputs[-1]` of the `DFTPlusTB` coincides with the `outputs[-1]` of the TB task (`Outputs` section).
    Note 3: the `outputs[-1]` of the DFT task is used as `inputs[0]` of the TB task.

    The archive.workflow2 section is:
        - name = 'DFT+TB'
        - method = DFTPlusTBMethod(
            dft_method_ref=dft_archive.data.model_method[-1],
            tb_method_ref=tb_archive.data.model_method[-1],
        )
        - inputs = [
            Link(name='Input Model System', section=dft_archive.data.model_system[0]),
        ]
        - outputs = [
            Link(name='Output TB Data', section=tb_archive.data.outputs[-1]),
        ]
        - tasks = [
            TaskReference(
                name='DFT SinglePoint Task',
                task=dft_archive.workflow2
                inputs=[
                    Link(name='Input Model System', section=dft_archive.data.model_system[0]),
                ],
                outputs=[
                    Link(name='Output DFT Data', section=dft_archive.data.outputs[-1]),
                ]
            ),
            TaskReference(
                name='TB SinglePoint Task',
                task=tb_archive.workflow2,
                inputs=[
                    Link(name='Output DFT Data', section=dft_archive.data.outputs[-1]),
                ],
                outputs=[
                    Link(name='Output tb Data', section=tb_archive.data.outputs[-1]),
                ]
            ),
        ]
    """

    @check_n_tasks(n_tasks=2)
    def link_task_inputs_outputs(self, tasks: list[TaskReference]) -> None:
        dft_task = tasks[0]
        tb_task = tasks[1]

        # Initial check
        if not dft_task.m_xpath('task.outputs'):
            return None

        # Input of DFT Task is the ModelSystem
        dft_task.inputs = [
            Link(name='Input Model System', section=self.inputs[0]),
        ]
        # Output of DFT Task is the output section of the DFT entry
        dft_task.outputs = [
            Link(name='Output DFT Data', section=dft_task.task.outputs[-1]),
        ]
        # Input of TB Task is the output of the DFT task
        tb_task.inputs = [
            Link(name='Output DFT Data', section=dft_task.task.outputs[-1]),
        ]
        # Output of TB Task is the output section of the TB entry
        tb_task.outputs = [
            Link(name='Output TB Data', section=self.outputs[-1]),
        ]

    # TODO check if implementing overwritting the FermiLevel.value in the TB entry from the DFT entry

    @check_n_tasks(n_tasks=2)
    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

        # Check if `tasks` are not SinglePoints
        for task in self.tasks:
            if not task.task:
                logger.error(
                    'A `DFTPlusTB` workflow must have two `SinglePoint` tasks references.'
                )
                return
            if not isinstance(task.task, 'SinglePoint'):
                logger.error(
                    'The referenced tasks in the `DFTPlusTB` workflow must be of type `SinglePoint`.'
                )
                return

        # Define name of the workflow
        self.name = 'DFT+TB'

        # Resolve `method`
        method_refs = self.resolve_method_refs(
            tasks=self.tasks,
            tasks_names=['DFT SinglePoint Task', 'TB SinglePoint Task'],
        )
        if method_refs is not None and len(method_refs) == 2:
            self.method = DFTPlusTBMethod(
                dft_method_ref=method_refs[0],
                tb_method_ref=method_refs[1],
            )

        # Resolve `tasks[*].inputs` and `tasks[*].outputs`
        self.link_task_inputs_outputs(tasks=self.tasks)
