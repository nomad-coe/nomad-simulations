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
# from nomad.metainfo import Environment
# from .run import Run
# from .calculation import Calculation
# from .method import Method
# from .system import System
# from . import run
# from . import method
# from . import calculation
# from . import system

# m_env = Environment()
# m_env.m_add_sub_section(Environment.packages, run.m_package)
# m_env.m_add_sub_section(Environment.packages, method.m_package)
# m_env.m_add_sub_section(Environment.packages, calculation.m_package)
# m_env.m_add_sub_section(Environment.packages, system.m_package)
import numpy as np

from nomad.datamodel.data import EntryData
from nomad.datamodel.metainfo.basesections import Simulation as BaseSimulation
from nomad.metainfo import SubSection
from .model_system import ModelSystem
from .model_method import ModelMethod
from .outputs import Outputs


class Simulation(BaseSimulation, EntryData):
    """ """

    # m_def = Section(extends_base_section=True)

    model_system = SubSection(sub_section=ModelSystem.m_def, repeats=True)

    model_method = SubSection(sub_section=ModelMethod.m_def, repeats=True)

    outputs = SubSection(sub_section=Outputs.m_def, repeats=True)

    def _set_system_branch_depth(
        self, system_parent: ModelSystem, branch_depth: int = 0
    ):
        for system_child in system_parent.model_system:
            system_child.branch_depth = branch_depth + 1
            self._set_system_branch_depth(system_child, branch_depth + 1)

    def normalize(self, archive, logger) -> None:
        super(EntryData, self).normalize(archive, logger)

        # Finding which is the representative system of a calculation: typically, we will
        # define it as the last system reported (CHECK THIS!).
        # TODO extend adding the proper representative system extraction using `normalizer.py`
        if self.model_system is None:
            logger.error("No system information reported.")
            return
        system_ref = self.model_system[-1]
        system_ref.is_representative = True
        self.m_cache["system_ref"] = system_ref

        # Setting up the `branch_depth` in the parent-child tree
        for system_parents in self.model_system:
            system_parents.branch_depth = 0
            if len(system_parents.model_system) == 0:
                continue
            self._set_system_branch_depth(system_parents)
