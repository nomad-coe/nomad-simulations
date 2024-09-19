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

# from nomad.datamodel.metainfo.workflow import Link, Task, Workflow
from nomad.metainfo import Quantity, SectionProxy, SubSection

from nomad_simulations.schema_packages.model_method import BaseModelMethod
from nomad_simulations.schema_packages.model_system import ModelSystem
from nomad_simulations.schema_packages.outputs import Outputs


class Link(ArchiveSection):
    name = Quantity(type=str)
    section = Quantity(
        type=ArchiveSection,
        description="""
        A reference to the section that contains the actual input or output data.
        """,
    )


class BaseWorkflow(ArchiveSection):
    name = Quantity(type=str)
    inputs = SubSection(sub_section=Link.m_def, repeats=True)
    outputs = SubSection(sub_section=Link.m_def, repeats=True)


class Workflow(BaseWorkflow):
    tasks = SubSection(sub_section=SectionProxy('Workflow'), repeats=True)


class WorkflowReference(BaseWorkflow):
    task_reference = SubSection(sub_section=BaseWorkflow.m_def, repeats=True)
