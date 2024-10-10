from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger

from nomad.datamodel.metainfo.workflow_new import LinkReference
from nomad.metainfo import Quantity
from nomad.utils import extract_section

from nomad_simulations.schema_packages.outputs import SCFOutputs
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
            LinkReference(name='Input Model System', section=archive.data.model_system[0]),
            LinkReference(name='Input Model Method', section=archive.data.model_method[-1]),
        ]
        - outputs = [
            LinkReference(name='Output Data', section=archive.data.outputs[-1]),
        ]
        - tasks = []
    """

    # ? is this necessary?
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
        input_model_system = extract_section(archive, ['data', 'model_system'])
        output = extract_section(archive, ['data', 'outputs'])
        if not input_model_system or not output:
            logger.warning(
                'Could not find the `ModelSystem` or `Outputs` section in the archive.data section of the SinglePoint entry.'
            )
            return
        self.inputs = [
            LinkReference(name='Input Model System', section=input_model_system),
        ]
        self.outputs = [LinkReference(name='Output Data', section=output)]
        # `ModelMethod` is optional when defining workflows like the `SinglePoint`
        input_model_method = extract_section(archive, ['data', 'model_method'])
        if input_model_method is not None:
            self.inputs.append(
                LinkReference(name='Input Model Method', section=input_model_method)
            )

        # Resolve the `n_scf_steps` if the output is of `SCFOutputs` type
        if isinstance(output, SCFOutputs):
            if output.scf_steps is not None and len(output.scf_steps) > 0:
                self.n_scf_steps = len(output.scf_steps)
