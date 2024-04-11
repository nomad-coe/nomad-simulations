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

from typing import Any, Optional

from nomad import utils
from nomad.datamodel.data import ArchiveSection
from nomad.metainfo import (
    Quantity,
    SubSection,
    SectionProxy,
    Reference,
    Section,
    Context,
    MEnum,
)
from nomad.metainfo.metainfo import DirectQuantity, Dimension, _placeholder_quantity
from nomad.datamodel.metainfo.basesections import Entity

from .variables import Variables
from .numerical_settings import SelfConsistency


# We add `logger` for the `PhysicalProperty.variables_shape` method
logger = utils.get_logger(__name__)


class PhysicalProperty(ArchiveSection):
    """
    A base section used to define the physical properties obtained in a simulation, experiment, or in a post-processing
    analysis. The main quantity of the `PhysicalProperty` is `value`, whose instantiation has to be overwritten in the derived classes
    when inheriting from `PhysicalProperty`. It also contains `rank`, to define the tensor rank of the physical property, and
    `variables`, to define the variables over which the physical property varies (see variables.py). This class can also store several
    string identifiers and quantities for referencing and establishing the character of a physical property.
    """

    name = Quantity(
        type=str,
        description="""
        Name of the physical property. Example: `'ElectronicBandGap'`.
        """,
    )

    source = Quantity(
        type=MEnum('simulation', 'measurement', 'analysis'),
        default='simulation',
        description="""
        Source of the physical property. This quantity is related with the `Activity` performed to obtain the physical
        property. Example: an `ElectronicBandGap` can be obtained from a `'simulation'` or in a `'measurement'`.
        """,
    )

    type = Quantity(
        type=str,
        description="""
        Type categorization of the physical property. Example: an `ElectronicBandGap` can be `'direct'`
        or `'indirect'`.
        """,
        # ! add more examples in the description to improve the understanding of this quantity
    )

    label = Quantity(
        type=str,
        description="""
        Label for additional classification of the physical property. Example: an `ElectronicBandGap`
        can be labeled as `'DFT'` or `'GW'` depending on the methodology used to calculate it.
        """,
        # ! add more examples in the description to improve the understanding of this quantity
    )

    rank = DirectQuantity(
        type=Dimension,
        shape=['0..*'],
        default=[],
        name='rank',
        description="""
        Rank of the tensor describing the physical property. This quantity is stored as a Dimension:
            - scalars (tensor rank 0) have `rank=[]` (`len(rank) = 0`),
            - vectors (tensor rank 1) have `rank=[a]` (`len(rank) = 1`),
            - matrices (tensor rank 2), have `rank=[a, b]` (`len(rank) = 2`),
            - etc.
        """,
    )

    variables = SubSection(sub_section=Variables.m_def, repeats=True)

    # * `value` must be overwritten in the derived classes defining its type, unit, and description
    value: Quantity = _placeholder_quantity

    entity_ref = Quantity(
        type=Entity,
        description="""
        Reference to the entity that the physical property refers to. Examples:
            - a simulated physical property might refer to the macroscopic system or instead of a specific atom in the unit
            cell. In the first case, `outputs.model_system_ref` (see outputs.py) will point to the `ModelSystem` section,
            while in the second case, `entity_ref` will point to `AtomsState` section (see atoms_state.py).
        """,
    )

    physical_property_ref = Quantity(
        type=Reference(SectionProxy('PhysicalProperty')),
        description="""
        Reference to the `PhysicalProperty` section from which the physical property was derived. If `physical_property_ref`
        is populated, the quantity `is_derived` is set to True via normalization.
        """,
    )

    is_derived = Quantity(
        type=bool,
        default=False,
        description="""
        Flag indicating whether the physical property is derived from other physical properties. We make
        the distinction between directly parsed and derived physical properties:
            - Directly parsed: the physical property is directly parsed from the simulation output files.
            - Derived: the physical property is derived from other physical properties. No extra numerical settings
                are required to calculate the physical property.
        """,
    )

    is_scf_converged = Quantity(
        type=bool,
        description="""
        Flag indicating whether the physical property is converged or not after a SCF process. This quantity is connected
        with `SelfConsistency` defined in the `numerical_settings.py` module.
        """,
    )

    self_consistency_ref = Quantity(
        type=SelfConsistency,
        description="""
        Reference to the `SelfConsistency` section that defines the numerical settings to converge the
        physical property (see numerical_settings.py).
        """,
    )

    @property
    def variables_shape(self) -> Optional[list]:
        """
        Shape of the variables over which the physical property varies. This is extracted from
        `Variables.n_grid_points` and appended in a list.

        Example, a physical property which varies with `Temperature` and `ElectricField` will
        return `variables_shape = [n_temperatures, n_electric_fields]`.

        Returns:
            (list): The shape of the variables over which the physical property varies.
        """
        if self.variables is not None:
            return [v.get_n_grid_points(v.grid_points, logger) for v in self.variables]
        return []

    @property
    def full_shape(self) -> list:
        """
        Full shape of the physical property. This quantity is calculated as a concatenation of the `variables_shape`
        and `rank`:

            `full_shape = variables_shape + rank`

        where `rank` is passed as an attribute of the `PhysicalProperty` and is related with the order of
        the tensor of `value`, and `variables_shape` is obtained from the property-decorated function `variables_shape()`
        and is related with the shapes of the `variables` over which the physical property varies.

        Example: a physical property which is a 3D vector and varies with `variables=[Temperature, ElectricField]`
        will have `rank=[3]`, `variables_shape=[n_temperatures, n_electric_fields]`, and thus
        `full_shape=[n_temperatures, n_electric_fields, 3]`.

        Returns:
            (list): The full shape of the physical property.
        """
        return self.variables_shape + self.rank

    def __init__(self, m_def: Section = None, m_context: Context = None, **kwargs):
        super().__init__(m_def, m_context, **kwargs)

        # initialize a `_new_value` quantity copying the main attrs from the `_value` quantity (`type`, `unit`,
        # `description`); this will then be used to setattr the `value` quantity to the `_new_value` one with the
        # correct `shape=_full_shape`
        for quant in self.m_def.quantities:
            if quant.name == 'value':
                self._new_value = Quantity(
                    type=quant.type,
                    unit=quant.unit,  # ? this can be moved to __setattr__
                    description=quant.description,
                )
                break

    def __setattr__(self, name: str, val: Any) -> None:
        # For the special case of `value`, its `shape` needs to be defined from `_full_shape`
        if name == 'value':
            # non-scalar or scalar `val`
            try:
                value_shape = list(val.shape)
            except AttributeError:
                value_shape = []

            if value_shape != self.full_shape:
                raise ValueError(
                    f'The shape of the stored `value` {value_shape} does not match the full shape {self.full_shape} '
                    f'extracted from the variables `n_grid_points` and the `shape` defined in `PhysicalProperty`.'
                )
            self._new_value.shape = self.full_shape
            self._new_value = val.magnitude * val.u
            return super().__setattr__(name, self._new_value)
        return super().__setattr__(name, val)

    def _is_derived(self) -> bool:
        """
        Resolves if the physical property is derived or not.

        Returns:
            (bool): The flag indicating whether the physical property is derived or not.
        """
        if self.physical_property_ref is not None:
            return True
        return False

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        # Resolve if the physical property `is_derived` or not from another physical property.
        self.is_derived = self._is_derived()
