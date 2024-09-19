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

from functools import wraps
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger

from nomad.datamodel.data import ArchiveSection
from nomad.datamodel.metainfo.workflow import Workflow
from nomad.metainfo import SubSection

from nomad_simulations.schema_packages.model_method import BaseModelMethod
from nomad_simulations.schema_packages.outputs import Outputs


def check_n_tasks(n_tasks: Optional[int] = None):
    """
    Check if the `tasks` of a workflow exist. If the `n_tasks` input specified, it checks whether `tasks`
    is of the same length as `n_tasks`.

    Args:
        n_tasks (Optional[int], optional): The length of the `tasks` needs to be checked if set to an integer. Defaults to None.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not self.tasks:
                return None
            if n_tasks is not None and len(self.tasks) != n_tasks:
                return None

            return func(self, *args, **kwargs)

        return wrapper

    return decorator


class SimulationWorkflow(Workflow):
    """
    A base section used to define the workflows of a simulation with references to specific `tasks`, `inputs`, and `outputs`. The
    normalize function checks the definition of these sections and sets the name of the workflow.

    A `SimulationWorkflow` will be composed of:
        - a `method` section containing methodological parameters used specifically during the workflow,
        - a list of `inputs` with references to the `ModelSystem` and, optionally, `ModelMethod` input sections,
        - a list of `outputs` with references to the `Outputs` section,
        - a list of `tasks` containing references to the activity `Simulation` used in the workflow,
    """

    method = SubSection(
        sub_section=BaseModelMethod.m_def,
        description="""
        Methodological parameters used during the workflow.
        """,
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class BeyondDFTMethod(ArchiveSection):
    """
    An abstract section used to store references to the `ModelMethod` sections of each of the
    archives defining the `tasks` and used to build the standard `BeyondDFT` workflow. This section needs to be
    inherit and the method references need to be defined for each specific case (see, e.g., dft_plus_tb.py module).
    """

    pass


class BeyondDFT(SimulationWorkflow):
    """
    A base section used to represent a beyond-DFT workflow and containing a `method` section which uses references
    to the specific tasks `ModelMethod` sections.
    """

    method = SubSection(
        sub_section=BeyondDFTMethod.m_def,
        description="""
        Abstract sub section used to populate the `method` of a `BeyondDFT` workflow with references
        to the corresponding `SinglePoint` entries and their `ModelMethod` sections.
        """,
    )

    @check_n_tasks()
    def resolve_all_outputs(self) -> list[Outputs]:
        """
        Resolves all the `Outputs` sections from the `tasks` in the workflow. This is useful when
        the workflow is composed of multiple tasks and the outputs need to be stored in a list
        for further manipulation, e.g., to plot multiple band structures in a DFT+TB workflow.

        Returns:
            list[Outputs]: A list of all the `Outputs` sections from the `tasks`.
        """
        # Populate the list of outputs from the last element in `tasks`
        all_outputs = []
        for task in self.tasks:
            if not task.outputs:
                continue
            all_outputs.append(task.outputs[-1])
        return all_outputs

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


#     def resolve_beyonddft_method_ref(
#         self, task: Optional[Task]
#     ) -> Optional[BaseModelMethod]:
#         """
#         Resolves the `ModelMethod` reference for the `task`.

#         Args:
#             task (Task): The task to resolve the `ModelMethod` reference from.

#         Returns:
#             Optional[BaseModelMethod]: The resolved `ModelMethod` reference.
#         """
#         if not task or not task.inputs:
#             return None
#         for input in task.inputs:
#             if input.section is not None and isinstance(input.section, BaseModelMethod):
#                 return input.section
#         return None
