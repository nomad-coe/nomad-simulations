from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger

from nomad.datamodel.metainfo.workflow_new import LinkReference
from nomad.metainfo import Quantity
from nomad.utils import extract_section

from nomad_simulations.schema_packages.model_method import DFT, TB
from nomad_simulations.schema_packages.workflow import BeyondDFT, BeyondDFTMethod
from nomad_simulations.schema_packages.workflow.base_workflows import check_n_tasks

from .single_point import SinglePoint


class DFTPlusTBMethod(BeyondDFTMethod):
    """
    Section used to reference the `DFT` and `TB` `ModelMethod` sections in each of the archives
    conforming a DFT+TB simulation workflow.
    """

    dft_method_ref = Quantity(
        type=DFT,
        description="""
        Reference to the DFT `ModelMethod` section in the DFT task.
        """,
    )
    tb_method_ref = Quantity(
        type=TB,
        description="""
        Reference to the TB `ModelMethod` section in the TB task.
        """,
    )


class DFTPlusTB(BeyondDFT):
    """
    A base section used to represent a DFT+TB calculation workflow. The `DFTPlusTB` workflow is composed of
    two tasks: the initial DFT calculation + the final TB projection.

    The section only needs to be populated with (everything else is handled by the `normalize` function):
        i. The `tasks` as `TaskReference` sections, adding `task` to the specific `archive.workflow2` sections.

    Note 1: the `inputs[0]` of the `DFTPlusTB` coincides with the `inputs[0]` of the DFT task (`ModelSystem` section).
    Note 2: the `outputs[-1]` of the `DFTPlusTB` coincides with the `outputs[-1]` of the TB task (`Outputs` section).

    The archive.workflow2 section is:
        - name = 'DFT+TB'
        - method = DFTPlusTBMethod(
            dft_method_ref=dft_archive.data.model_method[-1],
            tb_method_ref=tb_archive.data.model_method[-1],
        )
        - inputs = [
            LinkReference(name='Input Model System', section=dft_archive.data.model_system[0]),
        ]
        - outputs = [
            LinkReference(name='Output TB Data', section=tb_archive.data.outputs[-1]),
        ]
        - tasks = [
            TaskReference(task=dft_archive.workflow2),
            TaskReference(task=tb_archive.workflow2),
        ]
    """

    @check_n_tasks(n_tasks=2)
    def resolve_inputs_outputs(self) -> None:
        """
        Resolve the `inputs` and `outputs` of the `DFTPlusTB` workflow.
        """
        # Input system reference
        inputs = extract_section(self.tasks[0], ['task', 'inputs'], full_list=True)
        if not inputs:
            return None
        # ! check if this works (maybe refs are wrong as the tb_archive and dftplustb_archive share the same mainfile)
        input_section = inputs[0].section
        self.inputs = [LinkReference(name='Input Model System', section=input_section)]

        # Output TB data reference
        outputs = extract_section(self.tasks[0], ['task', 'outputs'], full_list=True)
        if not outputs:
            return None
        # ! check if this works (maybe refs are wrong as the tb_archive and dftplustb_archive share the same mainfile)
        output_section = outputs[0].section
        self.outputs = [LinkReference(name='Output TB Data', section=output_section)]

    # TODO check if implementing overwritting the FermiLevel.value in the TB entry from the DFT entry

    @check_n_tasks(n_tasks=2)
    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

        # Check if `tasks` are not SinglePoints
        for task in self.tasks:
            if not task.task:
                logger.error(
                    'A `DFTPlusTB` workflow must have two `SinglePoint` tasks references.'
                )
                return
            if not isinstance(task.task, SinglePoint):
                logger.error(
                    'The referenced tasks in the `DFTPlusTB` workflow must be of type `SinglePoint`.'
                )
                return

        # Define name of the workflow
        self.name = 'DFT+TB'

        # Resolve `method`
        method_refs = self.resolve_method_refs(
            tasks=self.tasks,
            tasks_names=['DFT SinglePoint Task', 'TB SinglePoint Task'],
        )
        if method_refs is not None and len(method_refs) == 2:
            print(method_refs)
            self.method = DFTPlusTBMethod(
                dft_method_ref=method_refs[0], tb_method_ref=method_refs[1]
            )

        # Resolve `inputs` and `outputs` from the `tasks`
        self.resolve_inputs_outputs()
