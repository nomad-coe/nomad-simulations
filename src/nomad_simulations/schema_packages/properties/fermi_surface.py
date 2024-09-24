from typing import TYPE_CHECKING

import numpy as np
from nomad.metainfo import Quantity

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from nomad.metainfo import Context, Section
    from structlog.stdlib import BoundLogger

from nomad_simulations.schema_packages.physical_property import PhysicalProperty


# TODO This class is not implemented yet. @JosePizarro3 will work in another PR to implement it.
class FermiSurface(PhysicalProperty):
    """
    Energy boundary in reciprocal space that separates the filled and empty electronic states in a metal.
    It is related with the crossing points in reciprocal space by the chemical potential or, equivalently at
    zero temperature, the Fermi level.
    """

    iri = 'http://fairmat-nfdi.eu/taxonomy/FermiSurface'

    n_bands = Quantity(
        type=np.int32,
        description="""
        Number of bands / eigenvalues.
        """,
    )

    def __init__(
        self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        # ! `n_bands` need to be set up during initialization of the class
        self.rank = [int(kwargs.get('n_bands'))]
        self.name = self.m_def.name

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)
