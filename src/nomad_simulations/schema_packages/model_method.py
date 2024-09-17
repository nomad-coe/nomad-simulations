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

import re
from typing import TYPE_CHECKING, Optional

import numpy as np
from nomad.datamodel.data import ArchiveSection
from nomad.datamodel.metainfo.annotations import ELNAnnotation
from nomad.metainfo import URL, MEnum, Quantity, Section, SubSection

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from nomad.metainfo import Context
    from structlog.stdlib import BoundLogger

from nomad_simulations.schema_packages.atoms_state import CoreHole, OrbitalsState
from nomad_simulations.schema_packages.model_system import ModelSystem
from nomad_simulations.schema_packages.numerical_settings import NumericalSettings
from nomad_simulations.schema_packages.utils import is_not_representative


class BaseModelMethod(ArchiveSection):
    """
    A base section used to define the abstract class of a Hamiltonian section. This section is an
    abstraction of the `ModelMethod` section, which contains the settings and parameters used in
    the mathematical model solved in a simulation. This abstraction is needed in order to allow
    `ModelMethod` to be divided into specific `terms`, so that the total Hamiltonian is specified in
    `ModelMethod`, while its contributions are defined in `ModelMethod.terms`.

    Example: a custom model Hamiltonian containing two terms:
        $H = H_{V_{1}(r)}+H_{V_{2}(r)}$
    where $H_{V_{1}(r)}$ and $H_{V_{2}(r)}$ are the contributions of two different potentials written in
    real space coordinates $r$. These potentials could be defined in terms of a combination of parameters
    $(a_{1}, b_{1}, c_{1}...)$ for $V_{1}(r)$ and $(a_{2}, b_{2}, c_{2}...)$ for $V_{2}(r)$. If we name the
    total Hamiltonian as `'FF1'`:
        `ModelMethod.name = 'FF1'`
        `ModelMethod.contributions = [BaseModelMethod(name='V1', parameters=[a1, b1, c1]), BaseModelMethod(name='V2', parameters=[a2, b2, c2])]`

    Note: quantities such as `name`, `type`, `external_reference` should be descriptive enough so that the
    total Hamiltonian model or each of the terms or contributions can be identified.
    """

    normalizer_level = 1

    name = Quantity(
        type=str,
        description="""
        Name of the mathematical model. This is typically used to identify the model Hamiltonian used in the
        simulation. Typical standard names: 'DFT', 'TB', 'GW', 'BSE', 'DMFT', 'NMR', 'kMC'.
        """,
        a_eln=ELNAnnotation(component='StringEditQuantity'),
    )

    type = Quantity(
        type=str,
        description="""
        Identifier used to further specify the kind or sub-type of model Hamiltonian. Example: a TB
        model can be 'Wannier', 'DFTB', 'xTB' or 'Slater-Koster'. This quantity should be
        rewritten to a MEnum when inheriting from this class.
        """,
        a_eln=ELNAnnotation(component='StringEditQuantity'),
    )

    external_reference = Quantity(
        type=URL,
        description="""
        External reference to the model e.g. DOI, URL.
        """,
        a_eln=ELNAnnotation(component='URLEditQuantity'),
    )

    numerical_settings = SubSection(sub_section=NumericalSettings.m_def, repeats=True)

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class ModelMethod(BaseModelMethod):
    """
    A base section containing the mathematical model parameters. These are both the parameters of
    the model and the settings used in the simulation. Optionally, this section can be decomposed
    in a series of contributions by storing them under the `contributions` quantity.
    """

    contributions = SubSection(
        sub_section=BaseModelMethod.m_def,
        repeats=True,
        description="""
        Contribution or sub-term of the total model Hamiltonian.
        """,
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class ModelMethodElectronic(ModelMethod):
    """
    A base section used to define the parameters of a model Hamiltonian used in electronic structure
    calculations (TB, DFT, GW, BSE, DMFT, etc).
    """

    # ? Is this necessary or will it be defined in another way?
    is_spin_polarized = Quantity(
        type=bool,
        description="""
        If the simulation is done considering the spin degrees of freedom (then there are two spin
        channels, 'down' and 'up') or not.
        """,
    )

    # ? What about this quantity
    relativity_method = Quantity(
        type=MEnum(
            'scalar_relativistic',
            'pseudo_scalar_relativistic',
            'scalar_relativistic_atomic_ZORA',
        ),
        description="""
        Describes the relativistic treatment used for the calculation of the final energy
        and related quantities. If `None`, no relativistic treatment is applied.
        """,
        a_eln=ELNAnnotation(component='EnumEditQuantity'),
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class XCFunctional(ArchiveSection):
    """
    A base section used to define the parameters of an exchange or correlation functional.
    """

    libxc_name = Quantity(
        type=str,
        description="""
        Provides the name of one of the exchange or correlation (XC) functional following the libxc
        convention. For the code base containing the conventions, see https://gitlab.com/libxc/libxc.
        """,  # TODO: step away from the libxc naming convention
        a_eln=ELNAnnotation(component='StringEditQuantity'),
    )

    name = Quantity(
        type=MEnum('exchange', 'correlation', 'hybrid', 'contribution'),
        description="""
        Name of the XC functional. It can be one of the following: 'exchange', 'correlation',
        'hybrid', or 'contribution'.
        """,
    )

    weight = Quantity(
        type=np.float64,
        description="""
        Weight of the functional. This quantity is relevant when defining linear combinations of the
        different functionals. If not specified, its value is 1.
        """,
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )

    # ? add method to extract `name` from `libxc_name`

    def get_weight_name(self, weight: Optional[np.float64]) -> Optional[str]:
        """
        Returns the `weight` as a string with a "*" added at the end.

        Args:
            weight (Optional[np.float64]): The weight of the functional.

        Returns:
            (Optional[str]): The weight as a string with a "*" added at the end.
        """
        weight_name = ''
        if weight is not None:
            weight_name = f'{str(weight)}*'
        return weight_name

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

        # Appending `weight` as a string to `libxc_name`
        libxc_name_weight = ''
        if self.weight is not None:
            libxc_name_weight = self.get_weight_name(self.weight)
        if '*' not in self.libxc_name:
            self.libxc_name = libxc_name_weight + self.libxc_name

        # ! check with @ndaelman-hu if this makes sense (COMMENTED OUT FOR NOW)
        # Appending `"+alpha"` in `libxc_name` for hybrids in which the `exact_exchange_mixing_factor` is included
        # libxc_name_alpha = ''
        # if (
        #     self.name == 'hybrid'
        #     and 'exact_exchange_mixing_factor' in self.parameters.keys()
        # ):
        #     libxc_name_alpha = f'+alpha'
        # if '+alpha' not in self.libxc_name:
        #     self.libxc_name = self.libxc_name + libxc_name_alpha


class DFT(ModelMethodElectronic):
    """
    A base section used to define the parameters used in a density functional theory (DFT) calculation.
    """

    # ? Do we need to define `type` for DFT+U?

    jacobs_ladder = Quantity(
        type=MEnum('LDA', 'GGA', 'metaGGA', 'hyperGGA', 'hybrid', 'unavailable'),
        description="""
        Functional classification in line with Jacob's Ladder. See:
            - https://doi.org/10.1063/1.1390175 (original paper)
            - https://doi.org/10.1103/PhysRevLett.91.146401 (meta-GGA)
            - https://doi.org/10.1063/1.1904565 (hyper-GGA)
        """,
    )

    # ? This could be moved under `contributions`, @ndaelman-hu
    xc_functionals = SubSection(sub_section=XCFunctional.m_def, repeats=True)

    exact_exchange_mixing_factor = Quantity(
        type=np.float64,
        description="""
        Amount of exact exchange mixed in with the XC functional (value range = [0, 1]).
        """,
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )

    # ! MEnum this
    self_interaction_correction_method = Quantity(
        type=str,
        description="""
        Contains the name for the self-interaction correction (SIC) treatment used to
        calculate the final energy and related quantities. If skipped or empty, no special
        correction is applied.

        The following SIC methods are available:

        | SIC method                | Description                       |

        | ------------------------- | --------------------------------  |

        | `""`                      | No correction                     |

        | `"SIC_AD"`                | The average density correction    |

        | `"SIC_SOSEX"`             | Second order screened exchange    |

        | `"SIC_EXPLICIT_ORBITALS"` | (scaled) Perdew-Zunger correction explicitly on a
        set of orbitals |

        | `"SIC_MAURI_SPZ"`         | (scaled) Perdew-Zunger expression on the spin
        density / doublet unpaired orbital |

        | `"SIC_MAURI_US"`          | A (scaled) correction proposed by Mauri and co-
        workers on the spin density / doublet unpaired orbital |
        """,
        a_eln=ELNAnnotation(component='StringEditQuantity'),
    )

    van_der_waals_correction = Quantity(
        type=MEnum('TS', 'OBS', 'G06', 'JCHS', 'MDB', 'XC'),
        description="""
        Describes the Van der Waals (VdW) correction methodology. If `None`, no VdW correction is applied.

        | VdW method  | Reference                               |
        | --------------------- | ----------------------------------------- |
        | `"TS"`  | http://dx.doi.org/10.1103/PhysRevLett.102.073005 |
        | `"OBS"` | http://dx.doi.org/10.1103/PhysRevB.73.205101 |
        | `"G06"` | http://dx.doi.org/10.1002/jcc.20495 |
        | `"JCHS"` | http://dx.doi.org/10.1002/jcc.20570 |
        | `"MDB"` | http://dx.doi.org/10.1103/PhysRevLett.108.236402 and http://dx.doi.org/10.1063/1.4865104 |
        | `"XC"` | The method to calculate the VdW energy uses a non-local functional |
        """,
        a_eln=ELNAnnotation(component='EnumEditQuantity'),
    )

    def __init__(self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs):
        super().__init__(m_def, m_context, **kwargs)
        self._jacobs_ladder_map = {
            'lda': 'LDA',
            'gga': 'GGA',
            'mgg': 'meta-GGA',
            'hyb_mgg': 'hyper-GGA',
            'hyb': 'hybrid',
        }

    def resolve_libxc_names(
        self, xc_functionals: list[XCFunctional]
    ) -> Optional[list[str]]:
        """
        Resolves the `libxc_names` and sorts them from the list of `XCFunctional` sections.

        Args:
            xc_functionals (list[XCFunctional]): The list of `XCFunctional` sections.

        Returns:
            (Optional[list[str]]): The resolved and sorted `libxc_names`.
        """
        return sorted(
            [
                functional.libxc_name
                for functional in xc_functionals
                if functional.libxc_name is not None
            ]
        )

    def resolve_jacobs_ladder(
        self,
        libxc_names: list[str],
    ) -> str:
        """
        Resolves the `jacobs_ladder` from the `libxc_names`. The mapping (libxc -> NOMAD) is set in `self._jacobs_ladder_map`.

        Args:
            libxc_names (list[str]): The list of `libxc_names`.

        Returns:
            (str): The resolved `jacobs_ladder`.
        """
        if libxc_names is None:
            return 'unavailable'

        rung_order = {x: i for i, x in enumerate(self._jacobs_ladder_map.keys())}
        re_abbrev = re.compile(r'((HYB_)?[A-Z]{3})')

        abbrevs = []
        for xc_name in libxc_names:
            try:
                abbrev = re_abbrev.match(xc_name).group(1)
                abbrev = abbrev.lower() if abbrev == 'HYB_MGG' else abbrev[:3].lower()
                abbrevs.append(abbrev)
            except AttributeError:
                continue

        try:
            highest_rung_abbrev = (
                max(abbrevs, key=lambda x: rung_order[x]) if abbrevs else None
            )
        except KeyError:
            return 'unavailable'
        return self._jacobs_ladder_map.get(highest_rung_abbrev, 'unavailable')

    def resolve_exact_exchange_mixing_factor(
        self, xc_functionals: list[XCFunctional], libxc_names: list[str]
    ) -> Optional[float]:
        """
        Resolves the `exact_exchange_mixing_factor` from the `xc_functionals` and `libxc_names`.

        Args:
            xc_functionals (list[XCFunctional]): The list of `XCFunctional` sections.
            libxc_names (list[str]): The list of `libxc_names`.

        Returns:
            (Optional[float]): The resolved `exact_exchange_mixing_factor`.
        """

        for functional in xc_functionals:
            if functional.name == 'hybrid':
                return functional.parameters.get('exact_exchange_mixing_factor')

        def _scan_patterns(patterns: list[str], xc_name: str) -> bool:
            return any(x for x in patterns if re.search('_' + x + '$', xc_name))

        for xc_name in libxc_names:
            if not re.search('_XC?_', xc_name):
                continue
            if re.search('_B3LYP[35]?$', xc_name):
                return 0.2
            elif _scan_patterns(['HSE', 'PBEH', 'PBE_MOL0', 'PBE_SOL0'], xc_name):
                return 0.25
            elif re.search('_M05$', xc_name):
                return 0.28
            elif re.search('_PBE0_13$', xc_name):
                return 1 / 3
            elif re.search('_PBE38$', xc_name):
                return 3 / 8
            elif re.search('_PBE50$', xc_name):
                return 0.5
            elif re.search('_M06_2X$', xc_name):
                return 0.54
            elif _scan_patterns(['M05_2X', 'PBE_2X'], xc_name):
                return 0.56
        return None

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

        libxc_names = self.resolve_libxc_names(self.xc_functionals)
        if libxc_names is not None:
            # Resolves the `jacobs_ladder` from `libxc` mapping
            jacobs_ladder = self.resolve_jacobs_ladder(libxc_names)
            self.jacobs_ladder = (
                jacobs_ladder if self.jacobs_ladder is None else self.jacobs_ladder
            )

            # Resolves the `exact_exchange_mixing_factor` from the `xc_functionals` and `libxc_names`
            if self.xc_functionals is not None:
                exact_exchange_mixing_factor = (
                    self.resolve_exact_exchange_mixing_factor(
                        self.xc_functionals, libxc_names
                    )
                )
                self.exact_exchange_mixing_factor = (
                    exact_exchange_mixing_factor
                    if self.exact_exchange_mixing_factor is None
                    else self.exact_exchange_mixing_factor
                )


class TB(ModelMethodElectronic):
    """
    A base section containing the parameters pertaining to a tight-binding (TB) model calculation.
    The type of tight-binding model is specified in the `type` quantity.
    """

    type = Quantity(
        type=MEnum('DFTB', 'xTB', 'Wannier', 'SlaterKoster', 'unavailable'),
        default='unavailable',
        description="""
        Tight-binding model Hamiltonian type. The default is set to `'unavailable'` in case none of the
        standard types can be recognized. These can be:

        | Value | Reference |
        | --------- | ----------------------- |
        | `'DFTB'` | https://en.wikipedia.org/wiki/DFTB |
        | `'xTB'` | https://xtb-docs.readthedocs.io/en/latest/ |
        | `'Wannier'` | https://www.wanniertools.org/theory/tight-binding-model/ |
        | `'SlaterKoster'` | https://journals.aps.org/pr/abstract/10.1103/PhysRev.94.1498 |
        | `'unavailable'` | - |
        """,
        a_eln=ELNAnnotation(component='EnumEditQuantity'),
    )

    # ? these 2 quantities will change when `BasisSet` is defined
    n_orbitals = Quantity(
        type=np.int32,
        description="""
        Number of orbitals used as a basis to obtain the `TB` model.
        """,
    )

    orbitals_ref = Quantity(
        type=OrbitalsState,
        shape=['n_orbitals'],
        description="""
        References to the `OrbitalsState` sections that contain the orbitals information which are
        relevant for the `TB` model.

        Example: hydrogenated graphene with 3 atoms in the unit cell. The full list of `AtomsState` would
        be
            [
                AtomsState(chemical_symbol='C', orbitals_state=[OrbitalsState('s'), OrbitalsState('px'), OrbitalsState('py'), OrbitalsState('pz')]),
                AtomsState(chemical_symbol='C', orbitals_state=[OrbitalsState('s'), OrbitalsState('px'), OrbitalsState('py'), OrbitalsState('pz')]),
                AtomsState(chemical_symbol='H', orbitals_state=[OrbitalsState('s')]),
            ]

        The relevant orbitals for the TB model are the `'pz'` ones for each `'C'` atom. Then, we define:

            orbitals_ref= [OrbitalState('pz'), OrbitalsState('pz')]

        The relevant atoms information can be accessed from the parent AtomsState sections:
            atom_state = orbitals_ref[i].m_parent
            index = orbitals_ref[i].m_parent_index
            atom_position = orbitals_ref[i].m_parent.m_parent.positions[index]
        """,
    )

    def resolve_type(self) -> Optional[str]:
        """
        Resolves the `type` of the `TB` section if it is not already defined, and from the
        `m_def.name` of the section.

        Returns:
            (Optional[str]): The resolved `type` of the `TB` section.
        """
        return (
            self.m_def.name
            if self.m_def.name in ['DFTB', 'xTB', 'Wannier', 'SlaterKoster']
            else None
        )

    def resolve_orbital_references(
        self,
        model_systems: list[ModelSystem],
        logger: 'BoundLogger',
        model_index: int = -1,
    ) -> Optional[list[OrbitalsState]]:
        """
        Resolves the references to the `OrbitalsState` sections from the child `ModelSystem` section.

        Args:
            model_systems (list[ModelSystem]): The list of `ModelSystem` sections.
            logger (BoundLogger): The logger to log messages.
            model_index (int, optional): The `ModelSystem` section index from which resolve the references. Defaults to -1.

        Returns:
            Optional[list[OrbitalsState]]: The resolved references to the `OrbitalsState` sections.
        """
        try:
            model_system = model_systems[model_index]
        except IndexError:
            logger.warning(
                f'The `ModelSystem` section with index {model_index} was not found.'
            )
            return None

        # If `ModelSystem` is not representative, the normalization will not run
        if is_not_representative(model_system=model_system, logger=logger):
            return None

        # If `AtomicCell` is not found, the normalization will not run
        if not model_system.cell:
            logger.warning('`AtomicCell` section was not found.')
            return None
        atomic_cell = model_system.cell[0]

        # If there is no child `ModelSystem`, the normalization will not run
        atoms_state = atomic_cell.atoms_state
        model_system_child = model_system.model_system
        if not atoms_state or not model_system_child:
            logger.warning('No `AtomsState` or child `ModelSystem` section were found.')
            return None

        # We flatten the `OrbitalsState` sections from the `ModelSystem` section
        orbitals_ref = []
        for active_atom in model_system_child:
            # If the child is not an "active_atom", the normalization will not run
            if active_atom.type != 'active_atom':
                continue
            indices = active_atom.atom_indices
            for index in indices:
                try:
                    active_atoms_state = atoms_state[index]
                except IndexError:
                    logger.warning(
                        f'The `AtomsState` section with index {index} was not found.'
                    )
                    continue
                orbitals_state = active_atoms_state.orbitals_state
                for orbital in orbitals_state:
                    orbitals_ref.append(orbital)
        return orbitals_ref

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

        # Set `name` to "TB"
        self.name = 'TB'

        # Resolve `type` to be defined by the lower level class (Wannier, DFTB, xTB or SlaterKoster) if it is not already defined
        self.type = self.resolve_type()

        # Resolve `orbitals_ref` from the info in the child `ModelSystem` section and the `OrbitalsState` sections
        model_systems = self.m_xpath('m_parent.model_system', dict=False)
        if model_systems is None:
            logger.warning(
                'Could not find the `ModelSystem` sections. References to `OrbitalsState` will not be resolved.'
            )
            return
        # This normalization only considers the last `ModelSystem` (default `model_index` argument set to -1)
        orbitals_ref = self.resolve_orbital_references(
            model_systems=model_systems, logger=logger
        )
        if orbitals_ref is not None and len(orbitals_ref) > 0 and not self.orbitals_ref:
            self.n_orbitals = len(orbitals_ref)
            self.orbitals_ref = orbitals_ref


class Wannier(TB):
    """
    A base section used to define the parameters used in a Wannier tight-binding fitting.
    """

    is_maximally_localized = Quantity(
        type=bool,
        description="""
        If the projected orbitals are maximally localized or just a single-shot projection.
        """,
    )

    localization_type = Quantity(
        type=MEnum('single_shot', 'maximally_localized'),
        description="""
        Localization type of the Wannier orbitals.
        """,
    )

    n_bloch_bands = Quantity(
        type=np.int32,
        description="""
        Number of input Bloch bands to calculate the projection matrix.
        """,
    )

    energy_window_outer = Quantity(
        type=np.float64,
        unit='electron_volt',
        shape=[2],
        description="""
        Bottom and top of the outer energy window used for the projection.
        """,
    )

    energy_window_inner = Quantity(
        type=np.float64,
        unit='electron_volt',
        shape=[2],
        description="""
        Bottom and top of the inner energy window used for the projection.
        """,
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

        # Resolve `localization_type` from `is_maximally_localized`
        if self.localization_type is None:
            if self.is_maximally_localized is not None:
                if self.is_maximally_localized:
                    self.localization_type = 'maximally_localized'
                else:
                    self.localization_type = 'single_shot'


class SlaterKosterBond(ArchiveSection):
    """
    A base section used to define the Slater-Koster bond information betwee two orbitals.
    """

    orbital_1 = Quantity(
        type=OrbitalsState,
        description="""
        Reference to the first `OrbitalsState` section.
        """,
    )

    orbital_2 = Quantity(
        type=OrbitalsState,
        description="""
        Reference to the second `OrbitalsState` section.
        """,
    )

    # ? is this the best naming
    bravais_vector = Quantity(
        type=np.int32,
        default=[0, 0, 0],
        shape=[3],
        description="""
        The Bravais vector of the cell in 3 dimensional. This is defined as the vector that connects the
        two atoms that define the Slater-Koster bond. A bond can be defined between orbitals in the
        same unit cell (bravais_vector = [0, 0, 0]) or in neighboring cells (bravais_vector = [m, n, p] with m, n, p are integers).
        Default is [0, 0, 0].
        """,
    )

    # TODO add more names and in the table
    name = Quantity(
        type=MEnum('sss', 'sps', 'sds'),
        description="""
        The name of the Slater-Koster bond. The name is composed by the `l_quantum_symbol` of the orbitals
        and the cell index. Table of possible values:

        | Value   | `orbital_1.l_quantum_symbol` | `orbital_2.l_quantum_symbol` | `bravais_vector` |
        | ------- | ---------------------------- | ---------------------------- | ------------ |
        | `'sss'` | 's' | 's' | [0, 0, 0] |
        | `'sps'` | 's' | 'p' | [0, 0, 0] |
        | `'sds'` | 's' | 'd' | [0, 0, 0] |
        """,
    )

    # ? units
    integral_value = Quantity(
        type=np.float64,
        description="""
        The Slater-Koster bond integral value.
        """,
    )

    def __init__(self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs):
        super().__init__(m_def, m_context, **kwargs)
        # TODO extend this to cover all bond names
        self._bond_name_map = {
            'sss': ['s', 's', (0, 0, 0)],
            'sps': ['s', 'p', (0, 0, 0)],
            'sds': ['s', 'd', (0, 0, 0)],
        }

    def resolve_bond_name_from_references(
        self,
        orbital_1: Optional[OrbitalsState],
        orbital_2: Optional[OrbitalsState],
        bravais_vector: Optional[tuple],
        logger: 'BoundLogger',
    ) -> Optional[str]:
        """
        Resolves the `name` of the `SlaterKosterBond` from the references to the `OrbitalsState` sections.

        Args:
            orbital_1 (Optional[OrbitalsState]): The first `OrbitalsState` section.
            orbital_2 (Optional[OrbitalsState]): The second `OrbitalsState` section.
            bravais_vector (Optional[tuple]): The bravais vector of the cell.
            logger (BoundLogger): The logger to log messages.

        Returns:
            (Optional[str]): The resolved `name` of the `SlaterKosterBond`.
        """
        # Initial check
        if orbital_1 is None or orbital_2 is None:
            logger.warning('The `OrbitalsState` sections are not defined.')
            return None
        if bravais_vector is None:
            logger.warning('The `bravais_vector` is not defined.')
            return None

        # Check for `l_quantum_symbol` in `OrbitalsState` sections
        if orbital_1.l_quantum_symbol is None or orbital_2.l_quantum_symbol is None:
            logger.warning(
                'The `l_quantum_symbol` of the `OrbitalsState` bonds are not defined.'
            )
            return None

        bond_name = None
        value = [orbital_1.l_quantum_symbol, orbital_2.l_quantum_symbol, bravais_vector]
        # Check if `value` is found in the `self._bond_name_map` and return the key
        for key, val in self._bond_name_map.items():
            if val == value:
                bond_name = key
                break
        return bond_name

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

        # Resolve the SK bond `name` from the `OrbitalsState` references and the `bravais_vector`
        if self.orbital_1 and self.orbital_2 and self.bravais_vector is not None:
            if self.bravais_vector is not None:
                bravais_vector = tuple(self.bravais_vector)  # transformed for comparing
            self.name = self.resolve_bond_name_from_references(
                orbital_1=self.orbital_1,
                orbital_2=self.orbital_2,
                bravais_vector=bravais_vector,
                logger=logger,
            )


class SlaterKoster(TB):
    """
    A base section used to define the parameters used in a Slater-Koster tight-binding fitting.
    """

    bonds = SubSection(sub_section=SlaterKosterBond.m_def, repeats=True)

    overlaps = SubSection(sub_section=SlaterKosterBond.m_def, repeats=True)

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class xTB(TB):
    """
    A base section used to define the parameters used in an extended tight-binding (xTB) calculation.
    """

    # ? Deprecate this

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class Photon(ArchiveSection):
    """
    A base section used to define parameters of a photon, typically used for optical responses.
    """

    # TODO check other options and add specific refs
    multipole_type = Quantity(
        type=MEnum('dipolar', 'quadrupolar', 'NRIXS', 'Raman'),
        description="""
        Type used for the multipolar expansion: dipole, quadrupole, NRIXS, Raman, etc.
        """,
    )

    polarization = Quantity(
        type=np.float64,
        shape=[3],
        description="""
        Direction of the photon polarization in cartesian coordinates.
        """,
    )

    energy = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        Photon energy.
        """,
    )

    momentum_transfer = Quantity(
        type=np.float64,
        shape=[3],
        description="""
        Momentum transfer to the lattice. This quanitity is important for inelastic scatterings, like
        the ones happening in quadrupolar, Raman, or NRIXS processes.
        """,
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

        # Add warning in case `multipole_type` and `momentum_transfer` are not consistent
        if (
            self.multipole_type in ['quadrupolar', 'NRIXS', 'Raman']
            and self.momentum_transfer is None
        ):
            logger.warning(
                'The `Photon.momentum_transfer` is not defined but the `Photon.multipole_type` describes inelastic scattering processes.'
            )


class ExcitedStateMethodology(ModelMethodElectronic):
    """
    A base section used to define the parameters typical of excited-state calculations. "ExcitedStateMethodology"
    mainly refers to methodologies which consider many-body effects as a perturbation of the original
    DFT Hamiltonian. These are: GW, TDDFT, BSE.
    """  # Note: we don't really talk about Hamiltonians in DFT: their physics is accommodated in the functional itself

    n_states = Quantity(
        type=np.int32,
        description="""
        Number of states used to calculate the excitations.
        """,
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )

    n_empty_states = Quantity(
        type=np.int32,
        description="""
        Number of empty states used to calculate the excitations. This quantity is complementary to `n_states`.
        """,
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )

    broadening = Quantity(
        type=np.float64,
        unit='joule',
        description="""
        Lifetime broadening applied to the spectra in full-width at half maximum for excited-state calculations.
        """,
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class Screening(ExcitedStateMethodology):
    """
    A base section used to define the parameters that define the calculation of screening. This is usually done in
    RPA and linear response.
    """

    dielectric_infinity = Quantity(
        type=np.int32,
        description="""
        Value of the static dielectric constant at infinite q. For metals, this is infinite
        (or a very large value), while for insulators is finite.
        """,
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class GW(ExcitedStateMethodology):
    """
    A base section used to define the parameters of a GW calculation.
    """

    type = Quantity(
        type=MEnum(
            'G0W0',
            'scGW',
            'scGW0',
            'scG0W',
            'ev-scGW0',
            'ev-scGW',
            'qp-scGW0',
            'qp-scGW',
        ),
        description="""
        GW Hedin's self-consistency cycle:

        | Name      | Description                      | Reference             |
        | --------- | -------------------------------- | --------------------- |
        | `'G0W0'`  | single-shot                      | https://journals.aps.org/prb/abstract/10.1103/PhysRevB.74.035101 |
        | `'scGW'`  | self-consistent G and W               | https://journals.aps.org/prb/abstract/10.1103/PhysRevB.75.235102 |
        | `'scGW0'` | self-consistent G with fixed W0  | https://journals.aps.org/prb/abstract/10.1103/PhysRevB.54.8411 |
        | `'scG0W'` | self-consistent W with fixed G0  | -                     |
        | `'ev-scGW0'`  | eigenvalues self-consistent G with fixed W0   | https://journals.aps.org/prb/abstract/10.1103/PhysRevB.34.5390 |
        | `'ev-scGW'`  | eigenvalues self-consistent G and W   | https://journals.aps.org/prb/abstract/10.1103/PhysRevB.74.045102 |
        | `'qp-scGW0'`  | quasiparticle self-consistent G with fixed W0 | https://journals.aps.org/prb/abstract/10.1103/PhysRevB.76.115109 |
        | `'qp-scGW'`  | quasiparticle self-consistent G and W | https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.96.226402 |
        """,
        a_eln=ELNAnnotation(component='EnumEditQuantity'),
    )

    analytical_continuation = Quantity(
        type=MEnum(
            'pade',
            'contour_deformation',
            'ppm_GodbyNeeds',
            'ppm_HybertsenLouie',
            'ppm_vonderLindenHorsh',
            'ppm_FaridEngel',
            'multi_pole',
        ),
        description="""
        Analytical continuation approximations of the GW self-energy:

        | Name           | Description         | Reference                        |
        | -------------- | ------------------- | -------------------------------- |
        | `'pade'` | Pade's approximant  | https://link.springer.com/article/10.1007/BF00655090 |
        | `'contour_deformation'` | Contour deformation | https://journals.aps.org/prb/abstract/10.1103/PhysRevB.67.155208 |
        | `'ppm_GodbyNeeds'` | Godby-Needs plasmon-pole model | https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.62.1169 |
        | `'ppm_HybertsenLouie'` | Hybertsen and Louie plasmon-pole model | https://journals.aps.org/prb/abstract/10.1103/PhysRevB.34.5390 |
        | `'ppm_vonderLindenHorsh'` | von der Linden and P. Horsh plasmon-pole model | https://journals.aps.org/prb/abstract/10.1103/PhysRevB.37.8351 |
        | `'ppm_FaridEngel'` | Farid and Engel plasmon-pole model  | https://journals.aps.org/prb/abstract/10.1103/PhysRevB.47.15931 |
        | `'multi_pole'` | Multi-pole fitting  | https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.74.1827 |
        """,
        a_eln=ELNAnnotation(component='EnumEditQuantity'),
    )

    # TODO improve description
    interval_qp_corrections = Quantity(
        type=np.int32,
        shape=[2],
        description="""
        Band indices (in an interval) for which the GW quasiparticle corrections are calculated.
        """,
    )

    screening_ref = Quantity(
        type=Screening,
        description="""
        Reference to the `Screening` section that the GW calculation used to obtain the screened Coulomb interactions.
        """,
        a_eln=ELNAnnotation(component='ReferenceEditQuantity'),
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class BSE(ExcitedStateMethodology):
    """
    A base section used to define the parameters of a BSE calculation.
    """

    # ? does RPA relates with `screening_ref`?
    type = Quantity(
        type=MEnum('Singlet', 'Triplet', 'IP', 'RPA'),
        description="""
        Type of the BSE Hamiltonian solved:

            H_BSE = H_diagonal + 2 * gx * Hx - gc * Hc

        Online resources for the theory:
        - http://exciting.wikidot.com/carbon-excited-states-from-bse#toc1
        - https://www.vasp.at/wiki/index.php/Bethe-Salpeter-equations_calculations
        - https://docs.abinit.org/theory/bse/
        - https://www.yambo-code.eu/wiki/index.php/Bethe-Salpeter_kernel

        | Name | Description |
        | --------- | ----------------------- |
        | `'Singlet'` | gx = 1, gc = 1 |
        | `'Triplet'` | gx = 0, gc = 1 |
        | `'IP'` | Independent-particle approach |
        | `'RPA'` | Random Phase Approximation |
        """,
        a_eln=ELNAnnotation(component='EnumEditQuantity'),
    )

    solver = Quantity(
        type=MEnum('Full-diagonalization', 'Lanczos-Haydock', 'GMRES', 'SLEPc', 'TDA'),
        description="""
        Solver algotithm used to diagonalize the BSE Hamiltonian.

        | Name | Description | Reference |
        | --------- | ----------------------- | ----------- |
        | `'Full-diagonalization'` | Full diagonalization of the BSE Hamiltonian | - |
        | `'Lanczos-Haydock'` | Subspace iterative Lanczos-Haydock algorithm | https://doi.org/10.1103/PhysRevB.59.5441 |
        | `'GMRES'` | Generalized minimal residual method | https://doi.org/10.1137/0907058 |
        | `'SLEPc'` | Scalable Library for Eigenvalue Problem Computations | https://slepc.upv.es/ |
        | `'TDA'` | Tamm-Dancoff approximation | https://doi.org/10.1016/S0009-2614(99)01149-5 |
        """,
        a_eln=ELNAnnotation(component='EnumEditQuantity'),
    )

    screening_ref = Quantity(
        type=Screening,
        description="""
        Reference to the `Screening` section that the BSE calculation used to obtain the screened Coulomb interactions.
        """,
        a_eln=ELNAnnotation(component='ReferenceEditQuantity'),
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


# ? Is this class really necessary or should go in outputs.py?
class CoreHoleSpectra(ModelMethodElectronic):
    """
    A base section used to define the parameters used in a core-hole spectra calculation. This
    also contains reference to the specific methodological section (DFT, BSE) used to obtain the core-hole spectra.
    """

    m_def = Section(a_eln={'hide': ['type']})

    # # TODO add examples
    # solver = Quantity(
    #     type=str,
    #     description="""
    #     Solver algorithm used for the core-hole spectra.
    #     """,
    #     a_eln=ELNAnnotation(component="StringEditQuantity"),
    # )

    type = Quantity(
        type=MEnum('absorption', 'emission'),
        description="""
        Type of the CoreHole excitation spectra calculated, either "absorption" or "emission".
        """,
        a_eln=ELNAnnotation(component='EnumEditQuantity'),
    )

    edge = Quantity(
        type=MEnum(
            'K',
            'L1',
            'L2',
            'L3',
            'L23',
            'M1',
            'M2',
            'M3',
            'M23',
            'M4',
            'M5',
            'M45',
            'N1',
            'N2',
            'N3',
            'N23',
            'N4',
            'N5',
            'N45',
        ),
        description="""
        Edge label of the excited core-hole. This is obtained by normalization by using `core_hole_ref`.
        """,
    )

    core_hole_ref = Quantity(
        type=CoreHole,
        description="""
        Reference to the `CoreHole` section that contains the information of the edge of the excited core-hole.
        """,
        a_eln=ELNAnnotation(component='ReferenceEditQuantity'),
    )

    excited_state_method_ref = Quantity(
        type=ModelMethodElectronic,
        description="""
        Reference to the `ModelMethodElectronic` section (e.g., `DFT` or `BSE`) that was used to obtain the core-hole spectra.
        """,
        a_eln=ELNAnnotation(component='ReferenceEditQuantity'),
    )

    # TODO add normalization to obtain `edge`

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class DMFT(ModelMethodElectronic):
    """
    A base section used to define the parameters of a DMFT calculation.
    """

    m_def = Section(a_eln={'hide': ['type']})

    impurity_solver = Quantity(
        type=MEnum(
            'CT-INT',
            'CT-HYB',
            'CT-AUX',
            'ED',
            'NRG',
            'MPS',
            'IPT',
            'NCA',
            'OCA',
            'slave_bosons',
            'hubbard_I',
        ),
        description="""
        Impurity solver method used in the DMFT loop:

        | Name              | Reference                            |
        | ----------------- | ------------------------------------ |
        | `'CT-INT'`        | https://link.springer.com/article/10.1134/1.1800216 |
        | `'CT-HYB'`        | https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.97.076405 |
        | `'CT-AUX'`        | https://iopscience.iop.org/article/10.1209/0295-5075/82/57003 |
        | `'ED'`            | https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.72.1545 |
        | `'NRG'`           | https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.80.395 |
        | `'MPS'`           | https://journals.aps.org/prb/abstract/10.1103/PhysRevB.90.045144 |
        | `'IPT'`           | https://journals.aps.org/prb/abstract/10.1103/PhysRevB.45.6479 |
        | `'NCA'`           | https://journals.aps.org/prb/abstract/10.1103/PhysRevB.47.3553 |
        | `'OCA'`           | https://journals.aps.org/prb/abstract/10.1103/PhysRevB.47.3553 |
        | `'slave_bosons'`  | https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.57.1362 |
        | `'hubbard_I'`     | https://iopscience.iop.org/article/10.1088/0953-8984/24/7/075604 |
        """,
    )

    n_impurities = Quantity(
        type=np.int32,
        description="""
        Number of impurities mapped from the correlated atoms in the unit cell. This defines whether
        the DMFT calculation is done in a single-impurity or multi-impurity run.
        """,
    )

    n_orbitals = Quantity(
        type=np.int32,
        shape=['n_impurities'],
        description="""
        Number of correlated orbitals per impurity.
        """,
    )

    orbitals_ref = Quantity(
        type=OrbitalsState,
        shape=['n_orbitals'],
        description="""
        References to the `OrbitalsState` sections that contain the orbitals information which are
        relevant for the `DMFT` calculation.

        Example: hydrogenated graphene with 3 atoms in the unit cell. The full list of `AtomsState` would
        be
            [
                AtomsState(chemical_symbol='C', orbitals_state=[OrbitalsState('s'), OrbitalsState('px'), OrbitalsState('py'), OrbitalsState('pz')]),
                AtomsState(chemical_symbol='C', orbitals_state=[OrbitalsState('s'), OrbitalsState('px'), OrbitalsState('py'), OrbitalsState('pz')]),
                AtomsState(chemical_symbol='H', orbitals_state=[OrbitalsState('s')]),
            ]

        The relevant orbitals for the TB model are the `'pz'` ones for each `'C'` atom. Then, we define:

            orbitals_ref= [OrbitalState('pz'), OrbitalsState('pz')]

        The relevant impurities information can be accesed from the parent AtomsState sections:
            impurity_state = orbitals_ref[i].m_parent
            index = orbitals_ref[i].m_parent_index
            impurity_position = orbitals_ref[i].m_parent.m_parent.positions[index]
        """,
    )

    # ? Improve this with `orbitals_ref.occupation` and possibly a function?
    n_electrons = Quantity(
        type=np.float64,
        shape=['n_impurities'],
        description="""
        Initial number of valence electrons per impurity.
        """,
    )

    inverse_temperature = Quantity(
        type=np.float64,
        unit='1/joule',
        description="""
        Inverse temperature = 1/(kB*T).
        """,
    )

    # ? Check this once magnetic states are better covered in the schema. This will be probably under `ModelSystem`
    # ? by checking the spins in `AtomsState` for the `AtomicCell`
    # ! Check solid_dmft example by using magmom (atomic magnetic moments), and improve on AtomsState to include such moments
    magnetic_state = Quantity(
        type=MEnum('paramagnetic', 'ferromagnetic', 'antiferromagnetic'),
        description="""
        Magnetic state in which the DMFT calculation is done. This quantity can be obtained from
        `orbitals_ref` and their spin state.
        """,
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)
