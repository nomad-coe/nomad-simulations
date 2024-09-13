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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from nomad.datamodel.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger

import numpy as np
from nomad.config import config
from nomad.datamodel.data import Schema
from nomad.datamodel.metainfo.annotations import ELNAnnotation
from nomad.datamodel.metainfo.basesections import Activity, Entity
from nomad.metainfo import Datetime, Quantity, SchemaPackage, Section, SubSection

from nomad_simulations.schema_packages.model_method import ModelMethod
from nomad_simulations.schema_packages.model_system import ModelSystem
from nomad_simulations.schema_packages.outputs import Outputs
from nomad_simulations.schema_packages.utils import (
    get_composition,
    is_not_representative,
)

configuration = config.get_plugin_entry_point(
    'nomad_simulations.schema_packages:nomad_simulations_plugin'
)

m_package = SchemaPackage()


def set_not_normalized(func: 'Callable'):
    """
    Decorator to set the section as not normalized.
    Typically decorates the section initializer.
    """

    def wrapper(self, *args, **kwargs) -> None:
        func(self, *args, **kwargs)
        self._is_normalized = False

    return wrapper


def check_normalized(func: 'Callable'):
    """
    Decorator to check if the section is already normalized.
    Typically decorates the section normalizer.
    """

    def wrapper(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        if self._is_normalized:
            return None
        func(self, archive, logger)
        self._is_normalized = True

    return wrapper


class Program(Entity):
    """
    A base section used to specify a well-defined program used for computation.

    Synonyms:
     - code
     - software
    """

    name = Quantity(
        type=str,
        description="""
        The name of the program.
        """,
        a_eln=ELNAnnotation(component='StringEditQuantity'),
    )

    version = Quantity(
        type=str,
        description="""
        The version label of the program.
        """,
        a_eln=ELNAnnotation(component='StringEditQuantity'),
    )

    link = Quantity(
        type=str,
        description="""
        Website link to the program in published information.
        """,
        a_eln=ELNAnnotation(component='URLEditQuantity'),
    )

    version_internal = Quantity(
        type=str,
        description="""
        Specifies a program version tag used internally for development purposes.
        Any kind of tagging system is supported, including git commit hashes.
        """,
        a_eln=ELNAnnotation(component='StringEditQuantity'),
    )

    subroutine_name_internal = Quantity(
        type=str,
        description="""
        Specifies the name of the subroutine of the program at large.
        This only applies when the routine produced (almost) all of the output,
        so the naming is representative. This naming is mostly meant for users
        who are familiar with the program's structure.
        """,
        a_eln=ELNAnnotation(component='StringEditQuantity'),
    )

    compilation_host = Quantity(
        type=str,
        description="""
        Specifies the host on which the program was compiled.
        """,
        a_eln=ELNAnnotation(component='StringEditQuantity'),
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        pass


class BaseSimulation(Activity):
    """
    A computational simulation that produces output data from a given input model system
    and input methodological parameters.

    Synonyms:
        - computation
        - calculation
    """

    m_def = Section(
        links=['https://liusemweb.github.io/mdo/core/1.1/index.html#Calculation']
    )

    datetime_end = Quantity(
        type=Datetime,
        description="""
        The date and time when this computation ended.
        """,
        a_eln=ELNAnnotation(component='DateTimeEditQuantity'),
    )

    cpu1_start = Quantity(
        type=np.float64,
        unit='second',
        description="""
        The starting time of the computation on the (first) CPU 1.
        """,
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )

    cpu1_end = Quantity(
        type=np.float64,
        unit='second',
        description="""
        The end time of the computation on the (first) CPU 1.
        """,
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )

    wall_start = Quantity(
        type=np.float64,
        unit='second',
        description="""
        The internal wall-clock time from the starting of the computation.
        """,
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )

    wall_end = Quantity(
        type=np.float64,
        unit='second',
        description="""
        The internal wall-clock time from the end of the computation.
        """,
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )

    program = SubSection(sub_section=Program.m_def, repeats=False)

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        pass


class Simulation(BaseSimulation, Schema):
    """
    A `Simulation` is a computational calculation that produces output data from a given input model system
    and input (model) methodological parameters. The output properties obtained from the simulation are stored
    in a list under `outputs`.

    Each sub-section of `Simulation` is defined in their corresponding modules: `model_system.py`, `model_method.py`,
    and `outputs.py`.

    The basic entry data for a `Simulation`, known as `SinglePoint` workflow, contains all the self-consistent (SCF) steps
    performed to converge the calculation, i.e., we do not split each SCF step in its own entry but rather group them in a general one.

    Synonyms:
        - calculation
        - computation
    """

    model_system = SubSection(sub_section=ModelSystem.m_def, repeats=True)

    model_method = SubSection(sub_section=ModelMethod.m_def, repeats=True)

    outputs = SubSection(sub_section=Outputs.m_def, repeats=True)

    def _set_system_branch_depth(
        self, system_parent: ModelSystem, branch_depth: int = 0
    ):
        for system_child in system_parent.model_system:
            system_child.branch_depth = branch_depth + 1
            self._set_system_branch_depth(
                system_parent=system_child, branch_depth=branch_depth + 1
            )

    def resolve_composition_formula(self, system_parent: ModelSystem) -> None:
        """Determine and set the composition formula for `system_parent` and all of its
        descendants.

        Args:
            system_parent (ModelSystem): The upper-most level of the system hierarchy to consider.
        """

        def set_composition_formula(
            system: ModelSystem, subsystems: list[ModelSystem], atom_labels: list[str]
        ) -> None:
            """Determine the composition formula for `system` based on its `subsystems`.
            If `system` has no children, the atom_labels are used to determine the formula.

            Args:
                system (ModelSystem): The system under consideration.
                subsystems (list[ModelSystem]): The children of system.
                atom_labels (list[str]): The global list of atom labels corresponding
                to the atom indices stored in system.
            """
            if not subsystems:
                atom_indices = (
                    system.atom_indices if system.atom_indices is not None else []
                )
                subsystem_labels = (
                    [np.array(atom_labels)[atom_indices]]
                    if atom_labels
                    else ['Unknown' for atom in range(len(atom_indices))]
                )
            else:
                subsystem_labels = [
                    subsystem.branch_label
                    if subsystem.branch_label is not None
                    else 'Unknown'
                    for subsystem in subsystems
                ]
            if system.composition_formula is None:
                system.composition_formula = get_composition(
                    children_names=subsystem_labels
                )

        def get_composition_recurs(system: ModelSystem, atom_labels: list[str]) -> None:
            """Traverse the system hierarchy downward and set the branch composition for
            all (sub)systems at each level.

            Args:
                system (ModelSystem): The system to traverse downward.
                atom_labels (list[str]): The global list of atom labels corresponding
                to the atom indices stored in system.
            """
            subsystems = system.model_system
            set_composition_formula(
                system=system, subsystems=subsystems, atom_labels=atom_labels
            )
            if subsystems:
                for subsystem in subsystems:
                    get_composition_recurs(system=subsystem, atom_labels=atom_labels)

        atoms_state = (
            system_parent.cell[0].atoms_state if system_parent.cell is not None else []
        )
        atom_labels = (
            [atom.chemical_symbol for atom in atoms_state]
            if atoms_state is not None
            else []
        )
        get_composition_recurs(system=system_parent, atom_labels=atom_labels)

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super(Schema, self).normalize(archive, logger)

        # Finding which is the representative system of a calculation: typically, we will
        # define it as the last system reported (TODO CHECK THIS!).
        # TODO extend adding the proper representative system extraction using `normalizer.py`
        if not self.model_system:
            logger.error('No system information reported.')
            return
        system_ref = self.model_system[-1]
        # * We define is_representative in the parser
        # system_ref.is_representative = True
        self.m_cache['system_ref'] = system_ref

        # Setting up the `branch_depth` in the parent-child tree
        for system_parent in self.model_system:
            system_parent.branch_depth = 0
            if len(system_parent.model_system) == 0:
                continue
            self._set_system_branch_depth(system_parent=system_parent)

            if is_not_representative(model_system=system_parent, logger=logger):
                continue
            self.resolve_composition_formula(system_parent=system_parent)


m_package.__init_metainfo__()
