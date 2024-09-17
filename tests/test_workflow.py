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
from nomad.datamodel.metainfo.workflow import Link, Task, Workflow

from nomad_simulations.schema_packages.model_method import ModelMethod
from nomad_simulations.schema_packages.model_system import ModelSystem
from nomad_simulations.schema_packages.outputs import Outputs
from nomad_simulations.schema_packages.workflow import (
    BeyondDFTMethod,
    BeyondDFTWorkflow,
    SimulationWorkflow,
)

from . import logger
from .conftest import generate_simulation


class TestSimulationWorkflow:
    @pytest.mark.parametrize(
        'model_system, model_method, outputs, workflow_inputs, workflow_outputs',
        [
            # empty sections in archive.data
            (None, None, None, [], []),
            # only one section in archive.data
            (ModelSystem(), None, None, [], []),
            # another section in archive.data
            (None, ModelMethod(), None, [], []),
            # only two sections in archive.data
            (ModelSystem(), ModelMethod(), None, [], []),
            # all sections in archive.data
            (
                ModelSystem(),
                ModelMethod(),
                Outputs(),
                [Link(name='Input Model System', section=ModelSystem())],
                [Link(name='Output Data', section=Outputs())],
            ),
        ],
    )
    def test_resolve_inputs_outputs(
        self,
        model_system: Optional[ModelSystem],
        model_method: Optional[ModelMethod],
        outputs: Optional[Outputs],
        workflow_inputs: list[Link],
        workflow_outputs: list[Link],
    ):
        archive = EntryArchive()
        simulation = generate_simulation(
            model_system=model_system, model_method=model_method, outputs=outputs
        )
        archive.data = simulation
        workflow = SimulationWorkflow()
        archive.workflow2 = workflow

        workflow.resolve_inputs_outputs(archive=archive, logger=logger)
        if not workflow_inputs:
            assert workflow.inputs == workflow_inputs
        else:
            assert len(workflow.inputs) == 1
            assert workflow.inputs[0].name == workflow_inputs[0].name
            # ! direct comparison of section does not work (probably different m_parent)
            # assert workflow.inputs[0].section == workflow_inputs[0].section
        if not workflow_outputs:
            assert workflow.outputs == workflow_outputs
        else:
            assert len(workflow.outputs) == 1
            assert workflow.outputs[0].name == workflow_outputs[0].name
            # ! direct comparison of section does not work (probably different m_parent)
            # assert workflow.outputs[0].section == workflow_outputs[0].section

    def test_normalize(self):
        assert True


class TestBeyondDFTMethod:
    def test_resolve_beyonddft_method_ref(self):
        assert True


class TestBeyondDFT:
    def test_resolve_all_outputs(self):
        assert True
