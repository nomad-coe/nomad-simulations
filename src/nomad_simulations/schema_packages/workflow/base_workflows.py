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

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger

from nomad.datamodel.data import ArchiveSection
from nomad.datamodel.metainfo.workflow import Link, Task, Workflow
from nomad.metainfo import SubSection

from nomad_simulations.schema_packages.model_method import BaseModelMethod
from nomad_simulations.schema_packages.model_system import ModelSystem
from nomad_simulations.schema_packages.outputs import Outputs


class SimulationWorkflow(Workflow):
    """
    A base section used to define the workflows of a simulation with references to specific `tasks`, `inputs`, and `outputs`. The
    normalize function checks the definition of these sections and sets the name of the workflow.

    A `SimulationWorkflow` will be composed of:
        - a `method` section containing methodological parameters used specifically during the workflow,
        - a list of `inputs` with references to the `ModelSystem` or `ModelMethod` input sections,
        - a list of `outputs` with references to the `Outputs` section,
        - a list of `tasks` containing references to the activity `Simulation` used in the workflow,
    """

    method = SubSection(
        sub_section=BaseModelMethod.m_def,
        description="""Methodological parameters used during the workflow.""",
    )

    def resolve_inputs_outputs(
        self, archive: 'EntryArchive', logger: 'BoundLogger'
    ) -> None:
        """
        Resolves the `inputs` and `outputs` sections from the archive sections under `data` and stores
        them in private attributes.

        Args:
            archive (EntryArchive): The archive to resolve the sections from.
            logger (BoundLogger): The logger to log messages.
        """
        if (
            not archive.data.model_system
            or not archive.data.model_method
            or not archive.data.outputs
        ):
            logger.info(
                '`ModelSystem`, `ModelMethod` and `Outputs` required for normalization of `SimulationWorkflow`.'
            )
            return None
        self._input_systems = archive.data.model_system
        self._input_methods = archive.data.model_method
        self._outputs = archive.data.outputs

        # Resolve `inputs`
        if not self.inputs:
            self.m_add_sub_section(
                Workflow.inputs,
                Link(name='Input Model System', section=self._input_systems[0]),
            )
        # Resolve `outputs`
        if not self.outputs:
            self.m_add_sub_section(
                Workflow.outputs,
                Link(name='Output Data', section=self._outputs[-1]),
            )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

        # Resolve the `inputs` and `outputs` from the archive
        self.resolve_inputs_outputs(archive=archive, logger=logger)

        # Storing the initial `ModelSystem`
        for link in self.inputs:
            if isinstance(link.section, ModelSystem):
                self.initial_structure = link.section
                break


class BeyondDFTMethod(ArchiveSection):
    """
    An abstract section used to store references to the `ModelMethod` sections of each of the
    archives defining the `tasks` and used to build the standard workflow. This section needs to be
    inherit and the method references need to be defined for each specific case.
    """

    def resolve_beyonddft_method_ref(
        self, task: Optional[Task]
    ) -> Optional[BaseModelMethod]:
        """
        Resolves the `ModelMethod` reference for the `task`.

        Args:
            task (Task): The task to resolve the `ModelMethod` reference from.

        Returns:
            Optional[BaseModelMethod]: The resolved `ModelMethod` reference.
        """
        if not task or not task.inputs:
            return None
        for input in task.inputs:
            if input.section is not None and isinstance(input.section, BaseModelMethod):
                return input.section
        return None


class BeyondDFTWorkflow(SimulationWorkflow):
    method = SubSection(sub_section=BeyondDFTMethod.m_def)

    def resolve_all_outputs(self) -> list[Outputs]:
        """
        Resolves all the `Outputs` sections from the `tasks` in the workflow. This is useful when
        the workflow is composed of multiple tasks and the outputs need to be stored in a list
        for further manipulation, e.g., to plot multiple band structures in a DFT+TB workflow.

        Returns:
            list[Outputs]: A list of all the `Outputs` sections from the `tasks`.
        """
        all_outputs = []
        for task in self.tasks:
            all_outputs.append(task.outputs[-1])
        return all_outputs

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)
