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

import numpy as np

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger

from nomad.datamodel.metainfo.workflow import Link, Task
from nomad.metainfo import Quantity

from nomad_simulations.schema_packages.outputs import SCFOutputs
from nomad_simulations.schema_packages.workflow import SimulationWorkflow


class SinglePoint(SimulationWorkflow):
    """
    A `SimulationWorkflow` used to represent a single point calculation workflow. The `SinglePoint`
    workflow is the minimum workflow required to represent a simulation. The self-consistent steps of
    scf simulation are represented in the `SinglePoint` workflow.
    """

    n_scf_steps = Quantity(
        type=np.int32,
        description="""
        The number of self-consistent field (SCF) steps in the simulation.
        """,
    )

    def generate_task(self, archive: 'EntryArchive', logger: 'BoundLogger') -> Task:
        """
        Generates the `Task` section for the `SinglePoint` workflow with their `inputs` and `outputs`.

        Returns:
            Task: The generated `Task` section.
        """
        # Populate `_input_systems`, `_input_methods` and `_outputs`
        self._resolve_inputs_outputs_from_archive(archive=archive, logger=logger)

        # Generate the `Task` section
        task = Task()
        if self._input_systems:
            task.m_add_sub_section(
                Task.inputs,
                Link(name='Input Model System', section=self._input_systems[0]),
            )
        if self._input_methods:
            task.m_add_sub_section(
                Task.inputs,
                Link(name='Input Model Method', section=self._input_methods[0]),
            )
        if self._outputs:
            task.m_add_sub_section(
                Task.outputs,
                Link(name='Output Data', section=self._outputs[-1]),
            )
        return task

    def resolve_n_scf_steps(self) -> int:
        """
        Resolves the number of self-consistent field (SCF) steps in the simulation.

        Returns:
            int: The number of SCF steps.
        """
        for output in self.outputs:
            if not isinstance(output, SCFOutputs):
                continue
            if output.scf_steps is not None:
                return len(output.scf_steps)
        return 1

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

        if self.tasks is not None and len(self.tasks) > 1:
            logger.error('A `SinglePoint` workflow must have only one task.')
            return

        # Generate the `tasks` section if this does not exist
        if not self.tasks:
            task = self.generate_task(archive=archive, logger=logger)
            self.tasks.append(task)

        # Resolve `n_scf_steps`
        self.n_scf_steps = self.resolve_n_scf_steps()
