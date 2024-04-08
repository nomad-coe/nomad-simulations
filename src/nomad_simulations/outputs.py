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
from structlog.stdlib import BoundLogger

from nomad.datamodel.data import ArchiveSection
from nomad.metainfo import Quantity, SubSection, MEnum
from nomad.datamodel.metainfo.annotations import ELNAnnotation

from .model_system import ModelSystem
from .physical_property import PhysicalProperty


class ElectronicBandGap(PhysicalProperty):
    """ """

    shape = []

    type = Quantity(
        type=MEnum('direct', 'indirect'),
        description="""
        Type categorization of the electronic band gap. The electronic band gap can be `'direct'` or `'indirect'`.
        """,
    )

    value = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        The value of the electronic band gap.
        """,
    )

    # TODO add more functionalities here

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)


class Outputs(ArchiveSection):
    """
    Output properties of a simulation. This base class can be used for inheritance in any of the output properties
    defined in this schema.

    It contains references to the specific sections used to obtain the output properties, as well as
    information if the output `is_derived` from another output section or directly parsed from the simulation output files.
    """

    # TODO add time quantities

    normalizer_level = 2

    model_system_ref = Quantity(
        type=ModelSystem,
        description="""
        Reference to the `ModelSystem` section to which the output property references to and on
        on which the simulation is performed.
        """,
        a_eln=ELNAnnotation(component='ReferenceEditQuantity'),
    )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # List of properties
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    electronic_band_gap = SubSection(sub_section=ElectronicBandGap.m_def, repeats=True)

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        # Resolve if the output property `is_derived` or not.
        # self.is_derived = self.resolve_is_derived(self.outputs_ref)


class SCFOutputs(Outputs):
    """
    This section contains the self-consistent (SCF) steps performed to converge an output property.

    For simplicity, we contain the SCF steps of a simulation as part of the minimal workflow defined in NOMAD,
    the `SinglePoint`, i.e., we do not split each SCF step in its own entry. Thus, each `SinglePoint`
    `Simulation` entry in NOMAD contains the final output properties and all the SCF steps.
    """

    scf_steps = SubSection(
        sub_section=Outputs.m_def,
        repeats=True,
        description="""
        Self-consistent (SCF) steps performed for converging a given output property. Note that the SCF steps belong to
        the same minimal `Simulation` workflow entry which is known as `SinglePoint`.
        """,
    )

    def resolve_is_derived(self, logger: BoundLogger) -> bool:
        """
        Resolves if the physical property is converged or not after a SCF process.

        Returns:
            (bool): The flag indicating whether the physical property is converged or not after a SCF process.
        """
        for property_name in self.m_def.all_sub_sections.keys():  # ! Check this
            # We get the property of `SCFOutputs` that we want to check
            prop = getattr(self, property_name)

            # If it is None or does not contain a `self_consistency_ref` quantity we continue
            if prop is None:
                continue
            scf_parameters = prop.self_consistency_ref
            if scf_parameters is None:
                continue

            # We check the last 2 `scf_steps` values of such property
            scf_steps_convergence = self.scf_steps[:-2]
            scf_props = []
            for scf_step in scf_steps_convergence:
                scf_prop = getattr(scf_step, property_name)
                if scf_prop is None:
                    logger.warning(f'The `scf_step.{property_name}` does not exist.')
                    break
                if scf_prop.value.unit != scf_parameters.threshold_unit:
                    logger.error(
                        f'The units of the `scf_step.{property_name}.value` does not coincide with the units of the `self_consistency_ref.threshold_unit`.'
                    )
                    return
                scf_props.append(scf_prop.value.magnitude)  # we use only the magnitude

            # We check if the property `is_scf_converged` checking its difference with respect to `self_consistency_ref.threshold_change`
            threshold = scf_parameters.threshold_change
            scf_diff = abs(scf_props[0] - scf_props[1])
            if scf_diff <= threshold:
                prop.is_scf_converged = True
            else:
                logger.info(
                    f'The SCF process for the property `{property_name}` did not converge.'
                )
                prop.is_scf_converged = False

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        # Resolve the `is_scf_converged` flag for all SCF obtained properties
        self.resolve_is_derived(logger)
