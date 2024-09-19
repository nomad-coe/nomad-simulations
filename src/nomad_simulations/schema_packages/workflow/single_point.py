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
from nomad_simulations.schema_packages.utils import extract_all_simulation_subsections
from nomad_simulations.schema_packages.workflow import SimulationWorkflow


class SinglePoint(SimulationWorkflow):
    """
    A base section used to represent a single point calculation workflow. The `SinglePoint`
    workflow is the minimum workflow required to represent a simulation. The self-consistent steps of
    scf simulation are represented inside the `SinglePoint` workflow.

    The section only needs to be instantiated, and everything else will be extracted from the `normalize` function.
    The archive needs to have `archive.data` sub-sections (model_sytem, model_method, outputs) populated.

    The archive.workflow2 section is:
        - name = 'SinglePoint'
        - inputs = [
            Link(name='Input Model System', section=archive.data.model_system[0]),
            Link(name='Input Model Method', section=archive.data.model_method[-1]),
        ]
        - outputs = [
            Link(name='Output Data', section=archive.data.outputs[-1]),
        ]
        - tasks = []
    """

    n_scf_steps = Quantity(
        type=np.int32,
        default=1,
        description="""
        The number of self-consistent field (SCF) steps in the simulation. This is calculated
        in the normalizer by storing the length of the `SCFOutputs` section in archive.data. Defaults
        to 1.
        """,
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

        # Define name
        self.name = 'SinglePoint'

        # Define `inputs` and `outputs`
        input_model_system, input_model_method, output = (
            extract_all_simulation_subsections(archive=archive)
        )
        if not input_model_system or not input_model_method or not output:
            logger.warning(
                'Could not find the ModelSystem, ModelMethod, or Outputs section in the archive.data section of the SinglePoint entry.'
            )
            return
        self.inputs = [
            Link(name='Input Model System', section=input_model_system),
            Link(name='Input Model Method', section=input_model_method),
        ]
        self.outputs = [Link(name='Output Data', section=output)]

        # Resolve the `n_scf_steps` if the output is of `SCFOutputs` type
        if isinstance(output, SCFOutputs):
            if output.scf_steps is not None and len(output.scf_steps) > 0:
                self.n_scf_steps = len(output.scf_steps)
