from typing import TYPE_CHECKING

import numpy as np
from nomad.metainfo import Context, Quantity, Section, SubSection

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from nomad.metainfo import Context, Section
    from structlog.stdlib import BoundLogger

from nomad_simulations.schema_packages.physical_property import (
    PhysicalProperty,
    PropertyContribution,
)

##################
# Abstract classes
##################


class BaseForce(PhysicalProperty):
    """
    Abstract class used to define a common `value` quantity with the appropriate units
    for different types of forces, which avoids repeating the definitions for each
    force class.
    """

    value = Quantity(
        type=np.float64,
        shape=['*'],
        unit='newton',
        description="""
        """,
    )

    def __init__(
        self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.rank = [3]

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class ForceContribution(BaseForce, PropertyContribution):
    """
    Abstract class for incorporating specific force contributions to the `TotalForce`.
    The inheritance from `PropertyContribution` allows to link this contribution to a
    specific component (of class `BaseModelMethod`) of the over `ModelMethod` using the
    `model_method_ref` quantity.

    For example, for a force field calculation, the `model_method_ref` may point to a
    particular potential type (e.g., a Lennard-Jones potential between atom types X and Y),
    while for a DFT calculation, it may point to a particular electronic interaction term
    (e.g., 'XC' for the exchange-correlation term, or 'Hartree' for the Hartree term).
    Then, the contribution will be named according to this model component and the `value`
    quantity will contain the force contribution from this component evaluated over all
    relevant atoms or electrons or as a function of them.
    """

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


###################################
# List of specific force properties
###################################


class TotalForce(BaseForce):
    """
    The total force on a system. `contributions` specify individual force
    contributions to the `TotalForce`.
    """

    contributions = SubSection(sub_section=ForceContribution.m_def, repeats=True)

    def __init__(
        self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.name = self.m_def.name

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)
