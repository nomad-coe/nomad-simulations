from nomad.datamodel.data import ArchiveSection
from nomad.datamodel.datamodel import EntryArchive
from nomad.datamodel.metainfo.annotations import ELNAnnotation
from nomad.metainfo import MEnum, Quantity, SubSection
from nomad.units import ureg
import numpy as np
import pint
from scipy import constants as const
from structlog.stdlib import BoundLogger
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from nomad.metainfo import Context, Section

from nomad_simulations.schema_packages.atoms_state import AtomsState
from nomad_simulations.schema_packages.numerical_settings import (
    Mesh,
    NumericalSettings,
)
from nomad_simulations.schema_packages.properties.energies import EnergyContribution


class BasisSet(ArchiveSection):
    """A type section denoting a basis set component of a simulation.
    Should be used as a base section for more specialized sections.
    Allows for denoting the basis set's _scope_, i.e. to which entity it applies,
    e.g. atoms species, orbital type, Hamiltonian term.

    Examples include:
    - mesh-based basis sets, e.g. (projector-)(augmented) plane-wave basis sets
    - atom-centered basis sets, e.g. Gaussian-type basis sets, Slater-type orbitals, muffin-tin orbitals
    """

    species_scope = Quantity(
        type=AtomsState,
        shape=['*'],
        description="""
        Reference to the section `AtomsState` specifying the localization of the basis set.
        """,
        a_eln=ELNAnnotation(components='ReferenceEditQuantity'),
    )

    # TODO: add atom index-based instantiator for species if not present

    hamiltonian_scope = Quantity(
        type=EnergyContribution,
        shape=['*'],
        description="""
        Reference to the section `EnergyContribution` containing the information
        of the Hamiltonian term to which the basis set applies.
        """,
        a_eln=ELNAnnotation(components='ReferenceEditQuantity'),
    )

    # ? band_scope or orbital_scope: valence vs core

    def __init__(self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs):
        super().__init__(m_def, m_context, **kwargs)


class PlaneWaveBasisSet(BasisSet, Mesh):
    """
    Basis set over a reciprocal mesh, where each point $k_n$ represents a planar-wave basis function $\frac{1}{\sqrt{\omega}} e^{i k_n r}$.
    Typically the grid itself is cartesian with only points within a designated sphere considered.
    The cutoff radius may be defined by a reciprocal length, or more commonly, the equivalent kinetic energy for a free particle.

    * D. J. Singh and L. Nordström, \"Why Planewaves\" in Planewaves, pseudopotentials, and the LAPW method, 2nd ed. New York, NY: Springer, 2006, pp. 24-26.
    """

    cutoff_energy = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        Cutoff energy for the plane-wave basis set.
        The simulation uses plane waves with energies below this cutoff.
        """,
    )

    @property  # ? keep, or convert to `Quantity`?
    def cutoff_radius(self) -> pint.Quantity:
        """
        Compute the cutoff radius for the plane-wave basis set, expressed in reciprocal coordinates.
        """
        if self.cutoff_energy is None:
            return None
        m_e = const.m_e * ureg(const.unit('electron mass'))
        h = const.h * ureg(const.unit('Planck constant'))
        return np.sqrt(2 * m_e * self.cutoff_energy) / h

    def normalize(self, archive: EntryArchive, logger: BoundLogger) -> None:
        super(BasisSet, self).normalize(archive, logger)
        super(Mesh, self).normalize(archive, logger)


class APWPlaneWaveBasisSet(PlaneWaveBasisSet):
    """
    A `PlaneWaveBasisSet` specialized to the APW use case.
    Its main descriptors are defined in terms of the `MuffinTin` regions.
    """

    cutoff_fractional = Quantity(
        type=np.float64,
        shape=[],
        description="""
        The spherical cutoff parameter for the interstitial plane waves in the APW family.
        This cutoff has no units, referring to the product of the smallest muffin-tin radius
        and the length of the cutoff reciprocal vector ($r_{MT} * |K_{cut}|$).
        """,
    )

    def set_cutoff_fractional(
        self, mt_r_min: pint.Quantity, logger: BoundLogger
    ) -> None:
        """
        Compute the fractional cutoff parameter for the interstitial plane waves in the LAPW family.
        This parameter is defined wrt the smallest muffin-tin region.
        """
        reference_unit = 'angstrom'
        if self.cutoff_fractional is not None:
            logger.info(
                '`APWPlaneWaveBasisSet.cutoff_fractional` already defined. Will not overwrite.'
            )  #! extend implementation
            return
        elif self.cutoff_energy is None or mt_r_min is None:
            logger.warning(
                '`APWPlaneWaveBasisSet.cutoff_energy` and `APWPlaneWaveBasisSet.radius` must both be defined. Aborting normalization step.'
            )
            return
        self.cutoff_fractional = self.cutoff_radius.to(
            f'1 / {reference_unit}'
        ) * mt_r_min.to(reference_unit)


class AtomCenteredFunction(ArchiveSection):
    """
    Specifies a single function (term) in an atom-centered basis set.
    """

    pass

    # TODO: design system for writing basis functions like gaussian or slater orbitals


class AtomCenteredBasisSet(BasisSet):
    """
    Defines an atom-centered basis set.
    """

    functional_composition = SubSection(
        sub_section=AtomCenteredFunction.m_def, repeats=True
    )  # TODO change name

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)
        # self.name = self.m_def.name
        # TODO: set name based on basis functions
        # ? use naming BSE


class APWBaseOrbital(ArchiveSection):
    """Abstract base section for (S)(L)APW and local orbital component wavefunctions.
    It helps defining the interface with `APWLChannel`."""

    n_terms = Quantity(
        type=np.int32,
        description="""
        Number of terms in the local orbital.
        """,
    )

    energy_parameter = Quantity(
        type=np.float64,
        shape=['n_terms'],
        unit='joule',
        description="""
        Reference energy parameter for the augmented plane wave (APW) basis set.
        Is used to set the energy parameter for each state.
        """,
    )  # TODO: add approximation formula from energy parameter n

    energy_parameter_n = Quantity(
        type=np.int32,
        shape=['n_terms'],
        description="""
        Reference number of radial nodes for the augmented plane wave (APW) basis set.
        This is used to derive the `energy_parameter`.
        """,
    )

    energy_status = Quantity(
        type=MEnum('fixed', 'pre-optimization', 'post-optimization'),
        default='post-optimization',
        description="""
        Allow the code to optimize the initial energy parameter.
        """,
    )

    differential_order = Quantity(
        type=np.int32,
        shape=['n_terms'],
        description="""
        Derivative order of the radial wavefunction term.
        """,
    )  # TODO: add check non-negative # ? to remove

    def normalize(self, archive: EntryArchive, logger: BoundLogger) -> None:
        super().normalize(archive, logger)
        if (self.differential_order < 0).any():
            logger.error('`APWBaseOrbital.differential_order` must be non-negative.')


class APWOrbital(APWBaseOrbital):
    """
    Implementation of `APWWavefunction` capturing the foundational (S)(L)APW basis sets, all of the form $\sum_{lm} \left[ \sum_o c_{lmo} \frac{\partial}{\partial r}u_l(r, \epsilon_l) \right] Y_lm$.
    The energy parameter $\epsilon_l$ is always considered fixed during diagonalization, opposed to the original APW formulation.
    This representation then has to match the plane-wave $k_n$ points within the muffin-tin sphere.

    * D. J. Singh and L. Nordström, \"INTRODUCTION TO THE LAPW METHOD,\" in Planewaves, pseudopotentials, and the LAPW method, 2nd ed. New York, NY: Springer, 2006, pp. 43-52.
    """

    type = Quantity(
        type=MEnum('apw', 'lapw', 'slapw', 'spherical_dirac'),
        description="""
        Type of augmentation contribution. Abbreviations stand for:
        | name | description | radial product |
        |------|-------------|----------------|
        | APW  | augmented plane wave with a frozen energy parameter | $A_{lm, k_n} u_l (r, E_l)$ |
        | LAPW | linearized augmented plane wave with an optimized energy parameter | $A_{lm, k_n} u_l (r, E_l) + B_{lm, k_n} \dot{u}_{lm} (r, E_l^')$ |
        | SLAPW | super linearized augmented plane wave | -- |
        | spherical Dirac | spherical Dirac basis set | -- |

        * http://susi.theochem.tuwien.ac.at/lapw/
        """,
    )


class APWLocalOrbital(APWBaseOrbital):
    """
    Implementation of `APWWavefunction` capturing a local orbital extending a foundational APW basis set.
    Local orbitals allow for flexible additions to an `APWOrbital` specification.
    They may be included to describe semi-core states, virtual states, ghost bands, or improve overall convergence.

    * D. J. Singh and L. Nordström, \"Role of the Linearization Energies,\" in Planewaves, pseudopotentials, and the LAPW method, 2nd ed. New York, NY: Springer, 2006, pp. 49-52.
    """

    type = Quantity(
        type=MEnum('lo', 'LO', 'custom'),
        description="""
        Type of augmentation contribution. Abbreviations stand for:
        | name | description | radial product |
        |------|-------------|----------------|
        | lo   | 2-parameter local orbital | $A_l u_l (r, E_l) + B_l \dot{u}_l (r, E_l^')$ |
        | LO   | 3-parameter local orbital | $A_l u_l (r, E_l) + B_l \dot{u}_l (r, E_l^') + C_l \dot{u}_l (r, E_l^{''})$ |
        | custom | local orbital of a different formula |

        * http://susi.theochem.tuwien.ac.at/lapw/
        """,
    )

    boundary_order = Quantity(
        type=np.int32,
        shape=['n_terms'],
        description="""
        Differential order to which the radial wavefunction is matched at the boundary.
        """,
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)
        if np.any(np.isneginf(self.differential_orders)):
            logger.error('`APWLOrbital.differential_order` must be non-negative.')
        if np.any(np.isneginf(self.boundary_orders)):
            logger.error('`APWLOrbital.boundary_orders` must be non-negative.')


class APWLChannel(BasisSet):
    """
    Collection of all (S)(L)APW and local orbital components that contribute
    to a single $l$-channel. $l$ here stands for the angular momentum parameter
    in the Laplace spherical harmonics $Y_{l, m}$.
    """

    name = Quantity(
        type=np.int32,
        description="""
        Angular momentum quantum number of the local orbital.
        """,
    )  # TODO: add `l` as a quantity

    n_wavefunctions = Quantity(
        type=np.int32,
        description="""
        Number of wavefunctions in the l-channel, i.e. $(2l + 1) n_orbitals$.
        """,
    )

    orbitals = SubSection(sub_section=APWBaseOrbital.m_def, repeats=True)

    def _determine_apw(self, logger: BoundLogger) -> dict[str, int]:
        """
        Produce a count of the APW components in the l-channel.
        """
        count = {'apw': 0, 'lapw': 0, 'slapw': 0, 'lo': 0, 'other': 0}
        order_map = {0: 'apw', 1: 'lapw', 2: 'slapw'}  # TODO: add dirac?

        for orb in self.orbitals:
            err_msg = f'Unknown APW orbital type {orb.type} and order {orb.differential_order}.'
            if isinstance(orb, APWOrbital):
                if orb.type in order_map.values():
                    count[orb.type] += 1
                elif orb.differential_order in order_map:
                    count[order_map[orb.differential_order]] += 1
                elif orb.differential_order > 2:
                    count['slapw'] += 1
                else:
                    logger.warning(err_msg)
            elif isinstance(orb, APWLocalOrbital):
                count['lo'] += 1
            else:
                logger.warning(err_msg)
        return count

    def normalize(self, archive: EntryArchive, logger: BoundLogger) -> None:
        super().normalize(archive, logger)
        # self.name = self.m_def.name
        self.n_wavefunctions = len(self.orbitals) * (2 * self.name + 1)


class MuffinTinRegion(BasisSet, Mesh):
    """
    Muffin-tin region around atoms, containing the augmented part of the APW basis set.
    The latter is structured by l-channel. Each channel contains a base (S)(L)APW definition,
    which may be extended via local orbitals.
    """

    # there are 2 main ways of structuring the APW basis set
    # either as APW and lo in the MT region
    # or by l-channel in the MT region

    radius = Quantity(
        type=np.float64,
        shape=[],
        unit='meter',
        description="""
        The radius descriptor of the `MuffinTin` is spherical shape.
        """,
    )

    l_max = Quantity(
        type=np.int32,
        description="""
        Maximum angular momentum quantum number that is sampled.
        Starts at 0.
        """,
    )

    l_channels = SubSection(sub_section=APWLChannel.m_def, repeats=True)

    def _determine_apw(self, logger: BoundLogger) -> dict[str, int]:
        """
        Aggregate the APW component count in the muffin-tin region.
        """
        count = {'apw': 0, 'lapw': 0, 'slapw': 0, 'lo': 0, 'other': 0}
        for channel in self.l_channels:
            count.update(channel._determine_apw(logger))
        return count

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)
        # self.type = 'spherical'


class BasisSetContainer(NumericalSettings):
    """
    A section defining the full basis set used for representing the electronic structure
    during the diagonalization of a Hamiltonian (component), as defined in `ModelMethod`.
    This section may contain multiple basis set specifications under the basis_set_components,
    each with their own parameters.
    """

    native_tier = Quantity(
        type=str,  # to be overwritten by a parser `MEnum`
        description="""
        Code-specific tag indicating the overall precision based on the basis set parameters.
        The naming conventions of the same code are used. See the parser implementation for the possible values.
        The number of tiers varies, but a typical example would be `low`, `medium`, `high`.
        """,
    )

    # TODO: add reference to `electronic_structure`,
    # specifying to which electronic structure representation the basis set is applied
    # e.g. wavefunction, density, grids for subroutines, etc.

    basis_set_components = SubSection(sub_section=BasisSet.m_def, repeats=True)

    def _determine_apw(self, logger: BoundLogger) -> str:
        """
        Derive the basis set name for a (S)(L)APW case.
        """
        answer, has_plane_wave = '', False
        for comp in self.basis_set_components:
            if isinstance(comp, MuffinTinRegion):
                count = comp._determine_apw(logger)
                if count['apw'] + count['lapw'] + count['slapw'] > 0:
                    if count['slapw'] > 0:
                        answer += 'slapw'.upper()
                    elif count['lapw'] > 0:
                        answer += 'lapw'.upper()
                    elif count['apw'] > 0:
                        answer += 'apw'.upper()
                    if count['lo'] > 0:
                        answer += '+lo'
            elif isinstance(comp, PlaneWaveBasisSet):
                has_plane_wave = True
        return answer if has_plane_wave else ''

    def _smallest_mt(self) -> MuffinTinRegion:
        """
        Scan the container for the smallest muffin-tin region.
        """
        mt_min = None
        for comp in self.basis_set_components:
            if isinstance(comp, MuffinTinRegion):
                if mt_min is None or comp.radius < mt_min.radius:
                    mt_min = comp
        return mt_min

    def normalize(self, archive: EntryArchive, logger: BoundLogger) -> None:
        super().normalize(archive, logger)
        pws = [
            comp
            for comp in self.basis_set_components
            if isinstance(comp, PlaneWaveBasisSet)
        ]
        if len(pws) > 1:
            logger.warning('Multiple plane-wave basis sets found were found.')
        if name := self._determine_apw(logger):
            self.name = name  # TODO: set name based on basis sets
            try:
                pws[0].set_cutoff_fractional(self._smallest_mt(), logger)
            except (IndexError, AttributeError):
                logger.error(
                    'Expected a `APWPlaneWaveBasisSet` instance, but found none.'
                )
