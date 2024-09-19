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

from nomad.datamodel.metainfo.workflow import Link
from nomad.metainfo import Quantity, Reference

from nomad_simulations.schema_packages.model_method import BaseModelMethod
from nomad_simulations.schema_packages.properties import FermiLevel
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
        type=Reference(BaseModelMethod),
        description="""Reference to the DFT `ModelMethod` section in the DFT task.""",
    )
    tb_method_ref = Quantity(
        type=Reference(BaseModelMethod),
        description="""Reference to the GW `ModelMethod` section in the TB task.""",
    )


class DFTPlusTB(BeyondDFT):
    """
    DFT+TB workflow is composed of two tasks: the initial DFT calculation + the final TB projection. This
    workflow section is used to define the same energy reference for both the DFT and TB calculations, by
    setting it up to the DFT calculation. The structure of the workflow is:

        - `self.inputs[0]`: the initial `ModelSystem` section in the DFT entry,
        - `self.outputs[0]`: the outputs section in the TB entry,
        - `tasks[0]`:
            - `tasks[0].task` (TaskReference): the reference to the `SinglePoint` task in the DFT entry,
            - `tasks[0].inputs[0]`: the initial `ModelSystem` section in the DFT entry,
            - `tasks[0].outputs[0]`: the outputs section in the DFT entry,
        - `tasks[1]`:
            - `tasks[1].task` (TaskReference): the reference to the `SinglePoint` task in the TB entry,
            - `tasks[1].inputs[0]`: the outputs section in the DFT entry,
            - `tasks[1].outputs[0]`: the outputs section in the TB entry,
        - `method`: references to the `ModelMethod` sections in the DFT and TB entries.
    """

    @check_n_tasks(n_tasks=2)
    def resolve_method(self) -> DFTPlusTBMethod:
        """
        Resolves the `DFT` and `TB` `ModelMethod` references for the `tasks` in the workflow by using the
        `resolve_beyonddft_method_ref` method from the `BeyondDFTMethod` section.

        Returns:
            DFTPlusTBMethod: The resolved `DFTPlusTBMethod` section.
        """
        method = DFTPlusTBMethod()

        # Check if TaskReference exists for both tasks
        for task in self.tasks:
            if not task.task:
                return None

        # DFT method reference
        dft_method = method.resolve_beyonddft_method_ref(task=self.tasks[0].task)
        if dft_method is not None:
            method.dft_method_ref = dft_method

        # TB method reference
        tb_method = method.resolve_beyonddft_method_ref(task=self.tasks[1].task)
        if tb_method is not None:
            method.tb_method_ref = tb_method

        return method

    @check_n_tasks(n_tasks=2)
    def link_tasks(self) -> None:
        """
        Links the `outputs` of the DFT task with the `inputs` of the TB task.
        """
        # Initial checks on the `inputs` and `tasks[*].outputs`
        if not self.inputs:
            return None
        for task in self.tasks:
            if not task.m_xpath('task.outputs'):
                return None

        # Assign dft task `inputs` to the `self.inputs[0]`
        dft_task = self.tasks[0]
        dft_task.inputs = [
            Link(
                name='Input Model System',
                section=self.inputs[0],
            )
        ]
        # and rewrite dft task `outputs` and its name
        dft_task.outputs = [
            Link(
                name='Output DFT Data',
                section=dft_task.task.outputs[-1],
            )
        ]

        # Assign tb task `inputs` to the `dft_task.outputs[-1]`
        tb_task = self.tasks[1]
        tb_task.inputs = [
            Link(
                name='Output DFT Data',
                section=dft_task.task.outputs[-1],
            ),
        ]
        # and rewrite tb task `outputs` and its name
        tb_task.outputs = [
            Link(
                name='Output TB Data',
                section=tb_task.task.outputs[-1],
            )
        ]

    @check_n_tasks(n_tasks=2)
    def overwrite_fermi_level(self) -> None:
        """
        Overwrites the Fermi level in the TB calculation with the Fermi level from the DFT calculation.
        """
        # Check if the `outputs` of the DFT task exist
        dft_task = self.tasks[0]
        if not dft_task.outputs:
            self.link_tasks()

        # Check if the `fermi_levels` exist in the DFT output
        if not dft_task.m_xpath('outputs[-1].section'):
            return None
        dft_output = dft_task.outputs[-1].section
        if not dft_output.fermi_levels:
            return None
        fermi_level = dft_output.fermi_levels[-1]

        # Assign the Fermi level to the TB output
        tb_task = self.tasks[1]
        if not tb_task.m_xpath('outputs[-1].section'):
            return None
        tb_output = tb_task.outputs[-1].section
        # ? Does appending like this work creating information in the TB entry?
        tb_output.fermi_levels.append(FermiLevel(value=fermi_level.value))

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

        # Initial check for the number of tasks
        if not self.tasks or len(self.tasks) != 2:
            logger.error('A `DFTPlusTB` workflow must have two tasks.')
            return

        # Check if tasks are `SinglePoint`
        for task in self.tasks:
            if task.m_def.name != 'SinglePoint':
                logger.error(
                    'A `DFTPlusTB` workflow must have two `SinglePoint` tasks.'
                )
                return

        # Define names of the workflow and `tasks`
        self.name = 'DFT+TB'
        self.tasks[0].name = 'DFT SinglePoint'
        self.tasks[1].name = 'TB SinglePoint'

        # Resolve method refs for each task and store under `method`
        self.method = self.resolve_method()

        # Link the tasks
        self.link_tasks()

        # Overwrite the Fermi level in the TB calculation
        # ? test if overwritting works
        self.overwrite_fermi_level()
