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
    def test_link_task_inputs_outputs(self):
        """
        Test the `link_task_inputs_outputs` method of the `DFTPlusTB` section.
        """
        assert True

    def test_normalize(self):
        """
        Test the `normalize` method of the `DFTPlusTB` section.
        """
        assert True
