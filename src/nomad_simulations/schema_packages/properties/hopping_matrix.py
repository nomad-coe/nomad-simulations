from typing import TYPE_CHECKING

import numpy as np
from nomad.metainfo import Quantity

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from nomad.metainfo import Context, Section
    from structlog.stdlib import BoundLogger

from nomad_simulations.schema_packages.physical_property import PhysicalProperty


class HoppingMatrix(PhysicalProperty):
    """
    Transition probability between two atomic orbitals in a tight-binding model.
    """

    iri = 'http://fairmat-nfdi.eu/taxonomy/HoppingMatrix'

    n_orbitals = Quantity(
        type=np.int32,
        description="""
        Number of orbitals in the tight-binding model. The `entity_ref` reference is used to refer to
        the `OrbitalsState` section.
        """,
    )

    degeneracy_factors = Quantity(
        type=np.int32,
        shape=['*'],
        description="""
        Degeneracy of each Wigner-Seitz point.
        """,
    )

    value = Quantity(
        type=np.complex128,
        unit='joule',
        description="""
        Value of the hopping matrix in joules. The elements are complex numbers defined for each Wigner-Seitz point and
        each pair of orbitals; thus, `rank = [n_orbitals, n_orbitals]`. Note this contains also the onsite values, i.e.,
        it includes the Wigner-Seitz point (0, 0, 0), hence the `CrystalFieldSplitting` values.
        """,
    )

    def __init__(
        self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        # ! n_orbitals need to be set up during initialization of the class
        self.rank = [int(kwargs.get('n_orbitals')), int(kwargs.get('n_orbitals'))]
        self.name = self.m_def.name

    # TODO add normalization to extract DOS, band structure, etc, properties from `HoppingMatrix`

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class CrystalFieldSplitting(PhysicalProperty):
    """
    Energy difference between the degenerated orbitals of an ion in a crystal field environment.
    """

    iri = 'http://fairmat-nfdi.eu/taxonomy/CrystalFieldSplitting'

    n_orbitals = Quantity(
        type=np.int32,
        description="""
        Number of orbitals in the tight-binding model. The `entity_ref` reference is used to refer to
        the `OrbitalsState` section.
        """,
    )

    value = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        Value of the crystal field splittings in joules. This is the intra-orbital local contribution, i.e., the same orbital
        at the same Wigner-Seitz point (0, 0, 0).
        """,
    )

    def __init__(
        self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        # ! `n_orbitals` need to be set up during initialization of the class
        self.rank = [int(kwargs.get('n_orbitals'))]
        self.name = self.m_def.name

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)
