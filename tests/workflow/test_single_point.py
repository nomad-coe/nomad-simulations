from typing import Optional

import pytest
from nomad.datamodel import EntryArchive
from nomad.datamodel.metainfo.workflow import Link

from nomad_simulations.schema_packages.model_method import ModelMethod
from nomad_simulations.schema_packages.model_system import ModelSystem
from nomad_simulations.schema_packages.outputs import Outputs, SCFOutputs
from nomad_simulations.schema_packages.workflow import SinglePoint

from ..conftest import generate_simulation
from . import logger


class TestBeyondDFT:
    @pytest.mark.parametrize(
        'model_system, model_method, outputs, result_inputs, result_outputs, result_n_scf_steps',
        [
            # no task
            (None, None, None, [], [], 1),
            (ModelSystem(), None, None, [], [], 1),
            (ModelSystem(), ModelMethod(), None, [], [], 1),
            (
                ModelSystem(),
                ModelMethod(),
                Outputs(),
                [
                    Link(name='Input Model System', section=ModelSystem()),
                    Link(name='Input Model Method', section=ModelMethod()),
                ],
                [Link(name='Output Data', section=Outputs())],
                1,
            ),
            (
                ModelSystem(),
                ModelMethod(),
                SCFOutputs(),
                [
                    Link(name='Input Model System', section=ModelSystem()),
                    Link(name='Input Model Method', section=ModelMethod()),
                ],
                [Link(name='Output Data', section=SCFOutputs())],
                1,
            ),
            (
                ModelSystem(),
                ModelMethod(),
                SCFOutputs(scf_steps=[Outputs(), Outputs(), Outputs()]),
                [
                    Link(name='Input Model System', section=ModelSystem()),
                    Link(name='Input Model Method', section=ModelMethod()),
                ],
                [
                    Link(
                        name='Output Data',
                        section=SCFOutputs(scf_steps=[Outputs(), Outputs(), Outputs()]),
                    )
                ],
                3,
            ),
        ],
    )
    def test_resolve_all_outputs(
        self,
        model_system: Optional[ModelSystem],
        model_method: Optional[ModelMethod],
        outputs: Optional[Outputs],
        result_inputs,
        result_outputs,
        result_n_scf_steps: Optional[int],
    ):
        """
        Test the `resolve_all_outputs` method of the `BeyondDFT` section.
        """
        archive = EntryArchive()

        # Add `Simulation` to archive
        simulation = generate_simulation(
            model_system=model_system, model_method=model_method, outputs=outputs
        )
        archive.data = simulation

        # Add `SinglePoint` to archive
        workflow = SinglePoint()
        archive.workflow2 = workflow

        workflow.normalize(archive=archive, logger=logger)

        assert workflow.name == 'SinglePoint'
        if not result_inputs:
            assert workflow.inputs == result_inputs
            assert workflow.outputs == result_outputs
        else:
            # ! comparing directly does not work becasue one is a section, the other a reference
            for i, input in enumerate(workflow.inputs):
                assert input.name == result_inputs[i].name
                assert isinstance(input.section, type(result_inputs[i].section))
            assert workflow.outputs[0].name == result_outputs[0].name
            assert isinstance(
                workflow.outputs[0].section, type(result_outputs[0].section)
            )
        assert workflow.n_scf_steps == result_n_scf_steps
