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

import numpy as np
from typing import List
from structlog.stdlib import BoundLogger

from nomad.units import ureg
from nomad.metainfo import SubSection, Quantity, MEnum, Section, Datetime
from nomad.datamodel.metainfo.annotations import ELNAnnotation
from nomad.datamodel.data import EntryData
from nomad.datamodel.metainfo.basesections import Entity, Activity

from .model_system import ModelSystem
from .model_method import ModelMethod
from .outputs import Outputs
from .utils import is_not_representative, get_composition

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

    compilation_host = Quantity(
        type=str,
        description="""
        Specifies the host on which the program was compiled.
        """,
        a_eln=ELNAnnotation(component='StringEditQuantity'),
    )

    def normalize(self, archive, logger) -> None:
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

    def normalize(self, archive, logger) -> None:
        pass


class Simulation(BaseSimulation, EntryData):
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
            self._set_system_branch_depth(system_child, branch_depth + 1)

    def resolve_composition_formula(
        self, system_parent: ModelSystem, logger: BoundLogger
    ) -> None:
        """
        """
        def set_branch_composition(system: ModelSystem, subsystems: List[ModelSystem], atom_labels: List[str]) -> None:
            if not subsystems:
                atom_indices = system.atom_indices if system.atom_indices is not None else []
                subsystem_labels = [np.array(atom_labels)[atom_indices]] if atom_labels else ['Unknown' for atom in range(len(atom_indices))]
            else:
                subsystem_labels = [subsystem.branch_label if subsystem.branch_label is not None else "Unknown" for subsystem in subsystems]
            if system.composition_formula is None:
                system.composition_formula = get_composition(subsystem_labels)

        def traverse_system_recurs(system, atom_labels):
            subsystems = system.model_system
            set_branch_composition(system, subsystems, atom_labels)
            if subsystems:
                for subsystem in subsystems:
                    traverse_system_recurs(subsystem, atom_labels)

        atoms_state = system_parent.cell[0].atoms_state if system_parent.cell is not None else []
        atom_labels = [atom.chemical_symbol for atom in atoms_state] if atoms_state is not None else []
        traverse_system_recurs(system_parent, atom_labels)

    def normalize(self, archive, logger) -> None:
        super(EntryData, self).normalize(archive, logger)

        # Finding which is the representative system of a calculation: typically, we will
        # define it as the last system reported (TODO CHECK THIS!).
        # TODO extend adding the proper representative system extraction using `normalizer.py`
        if self.model_system is None:
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
            self._set_system_branch_depth(system_parent)

            if is_not_representative(system_parent, logger):
                return
            self.resolve_composition_formula(system_parent, logger)