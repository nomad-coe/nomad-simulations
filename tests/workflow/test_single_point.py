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

from nomad_simulations.schema_packages.model_method import BaseModelMethod, ModelMethod
from nomad_simulations.schema_packages.model_system import ModelSystem
from nomad_simulations.schema_packages.outputs import Outputs
from nomad_simulations.schema_packages.workflow import SinglePoint

from ..conftest import generate_simulation
from . import logger


class TestSinglePoint:
    @pytest.mark.parametrize(
        'model_system, model_method, outputs, result',
        [
            # empty sections in archive.data
            (None, None, None, Task()),
            # only one section in archive.data
            (ModelSystem(), None, None, Task()),
            # another section in archive.data
            (None, ModelMethod(), None, Task()),
            # only two sections in archive.data
            (ModelSystem(), ModelMethod(), None, Task()),
            # all sections in archive.data
            (
                ModelSystem(),
                ModelMethod(),
                Outputs(),
                Task(
                    inputs=[
                        Link(name='Input Model System', section=ModelSystem()),
                        Link(name='Input Model Method', section=ModelMethod()),
                    ],
                    outputs=[
                        Link(name='Output Data', section=Outputs()),
                    ],
                ),
            ),
        ],
    )
    def test_generate_task(
        self,
        model_system: Optional[ModelSystem],
        model_method: Optional[ModelMethod],
        outputs: Optional[Outputs],
        result: Task,
    ):
        """
        Test the `generate_task` method of the `SinglePoint` section.
        """
        archive = EntryArchive()
        simulation = generate_simulation(
            model_system=model_system, model_method=model_method, outputs=outputs
        )
        archive.data = simulation
        workflow = SinglePoint()
        archive.workflow2 = workflow

        single_point_task = workflow.generate_task(archive=archive, logger=logger)
        if not result.inputs:
            assert isinstance(single_point_task, Task)
            assert not single_point_task.inputs and not single_point_task.outputs
        else:
            assert single_point_task.inputs[0].name == result.inputs[0].name
            assert single_point_task.inputs[1].name == result.inputs[1].name
            assert single_point_task.outputs[0].name == result.outputs[0].name

    def test_resolve_n_scf_steps():
        """
        Test the `resolve_n_scf_steps` method of the `SinglePoint` section.
        """
        assert True

    def test_normalize():
        """
        Test the `normalize` method of the `SinglePoint` section.
        """
        assert True
