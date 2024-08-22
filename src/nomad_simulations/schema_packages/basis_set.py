import itertools
from collections.abc import Iterable
from typing import Any, Callable, Optional

import numpy as np
import pint
from nomad import utils
from nomad.datamodel.data import ArchiveSection
from nomad.datamodel.datamodel import EntryArchive
from nomad.datamodel.metainfo.annotations import ELNAnnotation
from nomad.metainfo import MEnum, Quantity, SubSection
from nomad.units import ureg
from scipy import constants as const
from structlog.stdlib import BoundLogger

from nomad_simulations.schema_packages.atoms_state import AtomsState
from nomad_simulations.schema_packages.numerical_settings import (
    Mesh,
    NumericalSettings,
)
from nomad_simulations.schema_packages.properties.energies import EnergyContribution

logger = utils.get_logger(__name__)


def check_normalized(func: Callable):
    """
    Decorator to check if the section is already normalized.
    """

    def wrapper(self, archive: EntryArchive, logger: BoundLogger) -> None:
        if self._is_normalized:
            return None
        func(self, archive, logger)
        self._is_normalized = True

    return wrapper


class BasisSet(NumericalSettings):
    """A type section denoting a basis set component of a simulation.
    Should be used as a base section for more specialized sections.
    Allows for denoting the basis set's _scope_, i.e. to which entity it applies,
    e.g. atoms species, orbital type, Hamiltonian term.

    Examples include:
    - mesh-based basis sets, e.g. (projector-)(augmented) plane-wave basis sets
    - atom-centered basis sets, e.g. Gaussian-type basis sets, Slater-type orbitals, muffin-tin orbitals
    """

    name = Quantity(
        type=str,
        description="""
        Name of the basis set component.
        """,
    )

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

    def normalize(self, archive, logger):
        super().normalize(archive, logger)
        self.name = self.m_def.name


class PlaneWaveBasisSet(BasisSet, Mesh):
    """
    Basis set over a reciprocal mesh, where each point $k_n$ represents a planar-wave basis function $\frac{1}{\\sqrt{\\omega}} e^{i k_n r}$.
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
    """
    Abstract base section for (S)(L)APW and local orbital component wavefunctions.
    It helps defining the interface with `APWLChannel`.
    """

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

    def _get_open_quantities(self) -> set[str]:
        """Extract the open quantities of the `APWBaseOrbital`."""
        return {
            k for k, v in self.m_def.all_quantities.items() if self.m_get(v) is not None
        }

    def _get_lengths(self, quantities: set[str]) -> list[int]:
        """Extract the lengths of the `quantities` contained in the set."""
        present_quantities = set(quantities) & self._get_open_quantities()
        return [len(getattr(self, quant)) for quant in present_quantities]

    def _of_equal_length(self, lengths: list[int]) -> bool:
        """Check if all elements in the list are of equal length."""
        if len(lengths) == 0:
            return True
        else:
            ref_length = lengths[0]
            return all(length == ref_length for length in lengths)

    def get_n_terms(
        self,
        representative_quantities: set[str] = {
            'energy_parameter',
            'energy_parameter_n',
            'differential_order',
        },
    ) -> Optional[int]:
        """Determine the value of `n_terms` based on the lengths of the representative quantities."""
        lengths = self._get_lengths(representative_quantities)
        if not self._of_equal_length(lengths) or len(lengths) == 0:
            return None
        else:
            return lengths[0]

    def _check_non_negative(self, quantity_names: set[str]) -> bool:
        """Check if all elements in the set are non-negative."""
        for quantity_name in quantity_names:
            if isinstance(quant := self.get(quantity_name), Iterable):
                if np.any(np.array(quant) > 0):
                    return False
        return True

    @check_normalized
    def normalize(self, archive: EntryArchive, logger: BoundLogger) -> None:
        super().normalize(archive, logger)

        # enforce quantity length (will be used for type assignment)
        new_n_terms = self.get_n_terms()
        if self.n_terms is None:
            self.n_terms = new_n_terms
        elif self.n_terms != new_n_terms:
            if logger is not None:
                logger.error(
                    f'Inconsistent lengths of `APWBaseOrbital` quantities: {self.m_def.quantities}. Setting back to `None`.'
                )
            self.n_terms = None

        # enforce differential order constraints
        for quantity in ('differential_order', 'energy_parameter_n'):
            if self._check_non_negative({quantity}):
                quantity = None  # ? appropriate
                if logger is not None:
                    logger.error(
                        f'`{self.m_def}.{quantity}` must be completely non-negative. Resetting to `None`.'
                    )

        # use the differential order as naming convention
        self.name = (
            f'{sorted(self.differential_order)}'
            if len(self.differential_order) > 0
            else 'APW-like'
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # it's hard to enforce commutative diagrams between `_determine_apw` and `normalize`
        # instead, make all `_determine_apw` soft-coupled and dependent on the normalized state
        # leverage normalize ∘ normalize = normalize
        self._is_normalized = False


class APWOrbital(APWBaseOrbital):
    """
    Implementation of `APWWavefunction` capturing the foundational (S)(L)APW basis sets, all of the form $\\sum_{lm} \\left[ \\sum_o c_{lmo} \frac{\\partial}{\\partial r}u_l(r, \\epsilon_l) \right] Y_lm$.
    The energy parameter $\\epsilon_l$ is always considered fixed during diagonalization, opposed to the original APW formulation.
    This representation then has to match the plane-wave $k_n$ points within the muffin-tin sphere.

    Its `name` is showcased as `(s)(l)apw: <differential order>`.

    * D. J. Singh and L. Nordström, \"INTRODUCTION TO THE LAPW METHOD,\" in Planewaves, pseudopotentials, and the LAPW method, 2nd ed. New York, NY: Springer, 2006, pp. 43-52.
    """

    type = Quantity(
        type=MEnum('apw', 'lapw', 'slapw'),  # ? where to put 'spherical_dirac'
        description=r"""
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

    def do_to_type(self, do: Optional[list[int]]) -> Optional[str]:
        """
        Set the type of the APW orbital based on the differential order.
        """
        if do is None or len(do) == 0:
            return None

        do = sorted(do)
        if do == [0]:
            return 'apw'
        elif do == [0, 1]:
            return 'lapw'
        elif max(do) > 1:  # exciting definition
            return 'slapw'
        else:
            return None

    @check_normalized
    def normalize(self, archive: EntryArchive, logger: BoundLogger) -> None:
        super().normalize(archive, logger)
        # assign a APW orbital type
        # this snippet works of the previous normalization
        new_type = self.do_to_type(self.differential_order)
        if self.type is None:
            self.type = new_type
        elif self.type != new_type:
            logger.error(
                f'Inconsistent `APWOrbital` type: {self.type}. Setting back to `None`.'
            )
            self.type = None

        self.name = (
            f'{self.type.upper()}: {self.name}'
            if self.type and self.differential_order
            else self.name
        )


class APWLocalOrbital(APWBaseOrbital):
    """
    Implementation of `APWWavefunction` capturing a local orbital extending a foundational APW basis set.
    Local orbitals allow for flexible additions to an `APWOrbital` specification.
    They may be included to describe semi-core states, virtual states, ghost bands, or improve overall convergence.

    * D. J. Singh and L. Nordström, \"Role of the Linearization Energies,\" in Planewaves, pseudopotentials, and the LAPW method, 2nd ed. New York, NY: Springer, 2006, pp. 49-52.
    """

    # there's no community consensus on `type`

    @check_normalized
    def normalize(self, archive: EntryArchive, logger: BoundLogger) -> None:
        super().normalize(archive, logger)
        self.name = (
            f'LO: {sorted(self.differential_order)}'
            if self.differential_order
            else 'LO'
        )


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
    )

    n_orbitals = Quantity(
        type=np.int32,
        description="""
        Number of wavefunctions in the l-channel, i.e. $(2l + 1) n_orbitals$.
        """,
    )

    orbitals = SubSection(sub_section=APWBaseOrbital.m_def, repeats=True)

    def _determine_apw(self) -> dict[str, int]:
        """
        Produce a count of the APW components in the l-channel.
        Invokes `normalize` on `orbitals` to ensure the existence of `type`.
        """
        for orb in self.orbitals:
            orb.normalize(None, logger)

        type_count = {'apw': 0, 'lapw': 0, 'slapw': 0, 'lo': 0, 'other': 0}
        for orb in self.orbitals:
            if orb.type is None:
                type_count['other'] += 1
            elif isinstance(orb, APWOrbital) and orb.type.lower() in type_count.keys():
                type_count[orb.type] += 1
            elif isinstance(orb, APWLocalOrbital):
                type_count['lo'] += 1
            else:
                type_count['other'] += 1  # other de facto operates as a catch-all
        return type_count

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_normalized = False

    @check_normalized
    def normalize(self, archive: EntryArchive, logger: BoundLogger) -> None:
        ArchiveSection.normalize(self, archive, logger)
        self.n_orbitals = len(self.orbitals)


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

    def _determine_apw(self) -> dict[str, int]:
        """
        Aggregate the APW component count in the muffin-tin region.
        Invokes `normalize` on `l_channels`.
        """
        for l_channel in self.l_channels:
            l_channel.normalize(None, logger)

        type_count: dict[str, int] = {}
        if len(self.l_channels) > 0:
            # dynamically determine `type_count` structure
            for l_channel in self.l_channels:
                type_count.update(l_channel._determine_apw())
        return type_count

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_normalized = False

    @check_normalized
    def normalize(self, archive, logger):
        super().normalize(archive, logger)


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

    def _determine_apw(self) -> Optional[str]:
        """
        Derive the basis set name for a (S)(L)APW case, including local orbitals.
        Invokes `normalize` on `basis_set_components`.
        """
        answer, has_plane_wave = '', False
        for comp in self.basis_set_components:
            if isinstance(comp, MuffinTinRegion):
                comp.normalize(None, logger)
                type_count = comp._determine_apw()
                if sum([type_count[i] for i in ('apw', 'lapw', 'slapw')]) > 0:
                    if type_count['slapw'] > 0:
                        answer += 'slapw'.upper()
                    elif type_count['lapw'] > 0:
                        answer += 'lapw'.upper()
                    elif type_count['apw'] > 0:
                        answer += 'apw'.upper()
                elif type_count['lo'] > 0:
                    answer += '+lo'
                else:
                    answer = 'APW-like'
                    break
            elif isinstance(comp, PlaneWaveBasisSet):
                has_plane_wave = True
        return answer if has_plane_wave else None

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
        self.name = self._determine_apw()
        try:
            pws[0].set_cutoff_fractional(self._smallest_mt(), logger)
        except (IndexError, AttributeError):
            logger.error('Expected a `APWPlaneWaveBasisSet` instance, but found none.')


def generate_apw(
    species: dict[str, dict[str, Any]],
    cutoff: Optional[float] = None,
) -> BasisSetContainer:  # TODO: extend to cover all parsing use cases (maybe split up?)
    """
    Generate a mock APW basis set with the following structure:
    .
    ├── 1 x plane-wave basis set
    └── n x muffin-tin regions
        └── l_max x l-channels
            ├── orbitals
            └── local orbitals

    from a dictionary
    {
    <species_name>: {
            'r': <muffin-tin radius>,
            'l_max': <maximum angular momentum>,
            'orb_do': [[int]],
            'orb_param': [<APWOrbital.energy_parameter>|<APWOrbital.energy_parameter_n>],
            'lo_do': [[int]],
            'lo_param': [<APWOrbital.energy_parameter>|<APWOrbital.energy_parameter_n>],
        }
    }
    """

    basis_set_components: list[BasisSet] = []
    if cutoff is not None:
        pw = APWPlaneWaveBasisSet(cutoff_energy=cutoff)
        basis_set_components.append(pw)

    for sp_ref, sp in species.items():
        sp['r'] = sp.get('r', None)
        sp['l_max'] = sp.get('l_max', 0)
        sp['orb_d_o'] = sp.get('orb_d_o', [])
        sp['orb_param'] = sp.get('orb_param', [])
        sp['lo_d_o'] = sp.get('lo_d_o', [])
        sp['lo_param'] = sp.get('lo_param', [])

        basis_set_components.extend(
            [
                MuffinTinRegion(
                    species_scope=[sp_ref],
                    radius=sp['r'],
                    l_max=sp['l_max'],
                    l_channels=[
                        APWLChannel(
                            name=l_channel,
                            orbitals=list(
                                itertools.chain(
                                    (
                                        APWOrbital(
                                            energy_parameter=param,  # TODO: add energy_parameter_n
                                            differential_order=d_o,
                                        )
                                        for param, d_o in zip(
                                            sp['orb_param'], sp['orb_d_o']
                                        )
                                    ),
                                    (
                                        APWLocalOrbital(
                                            energy_parameter=param,  # TODO: add energy_parameter_n
                                            differential_order=d_o,
                                        )
                                        for param, d_o in zip(
                                            sp['lo_param'], sp['lo_d_o']
                                        )
                                    ),
                                )
                            ),
                        )
                        for l_channel in range(sp['l_max'] + 1)
                    ],
                )
            ]
        )

    return BasisSetContainer(basis_set_components=basis_set_components)
