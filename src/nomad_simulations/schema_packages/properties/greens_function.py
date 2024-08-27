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

import numpy as np
from nomad.metainfo import MEnum, Quantity

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from nomad.metainfo import Context, Section
    from structlog.stdlib import BoundLogger


from nomad_simulations.schema_packages.atoms_state import AtomsState, OrbitalsState
from nomad_simulations.schema_packages.physical_property import PhysicalProperty
from nomad_simulations.schema_packages.utils import get_variables
from nomad_simulations.schema_packages.variables import (
    Frequency,
    ImaginaryTime,
    KMesh,
    MatsubaraFrequency,
    Time,
    WignerSeitz,
)


class BaseGreensFunction(PhysicalProperty):
    """
    A base class used to define shared commonalities between Green's function-related properties. This is the case for `ElectronicGreensFunction`,
    `ElectronicSelfEnergy`, `HybridizationFunction` in DMFT simulations.

    These physical properties are matrices matrix represented in different spaces. These spaces are stored in
    `variables` and can be: `WignerSeitz` (real space), `KMesh`, `MatsubaraFrequency`, `Frequency`, `Time`, or `ImaginaryTime`.
    For example, G(k, ω) will have `variables = [KMesh(points), RealFrequency(points)]`.

    The `rank` is determined by the number of atoms and orbitals involved in correlations, so:
        `rank = [n_atoms, n_correlated_orbitals]`

    Further information in M. Wallerberger et al., Comput. Phys. Commun. 235, 2 (2019).
    """

    # ! we use `atoms_state_ref` and `orbitals_state_ref` to enforce order in the matrices

    n_atoms = Quantity(
        type=np.int32,
        description="""
        Number of atoms involved in the correlations effect and used for the matrix representation of the property.
        """,
    )

    atoms_state_ref = Quantity(
        type=AtomsState,
        shape=['n_atoms'],
        description="""
        Reference to the `AtomsState` section in which the Green's function properties are calculated.
        """,
    )

    n_correlated_orbitals = Quantity(
        type=np.int32,
        description="""
        Number of orbitals involved in the correlations effect and used for the matrix representation of the property.
        """,
    )

    correlated_orbitals_ref = Quantity(
        type=OrbitalsState,
        shape=['n_correlated_orbitals'],
        description="""
        Reference to the `OrbitalsState` section in which the Green's function properties are calculated.
        """,
    )

    spin_channel = Quantity(
        type=np.int32,
        description="""
        Spin channel of the corresponding electronic property. It can take values of 0 and 1.
        """,
    )

    local_model_type = Quantity(
        type=MEnum('impurity', 'lattice'),
        description="""
        Type of Green's function calculated from the mapping of the local Hubbard-Kanamori model
        into the Anderson impurity model.

        The `impurity` Green's function describe the electronic correlations for the impurity, and it
        is a local function. The `lattice` Green's function includes the coupling to the lattice
        and hence it is a non-local function. In DMFT, the `lattice` term is approximated to be the
        `impurity` one, so that these simulations are converged if both types of the local
        part of the `lattice` Green's function coincides with the `impurity` Green's function.
        """,
    )

    space_id = Quantity(
        type=MEnum(
            'r',
            'rt',
            'rw',
            'rit',
            'riw',
            'k',
            'kt',
            'kw',
            'kit',
            'kiw',
            't',
            'it',
            'w',
            'iw',
        ),
        description="""
        String used to identify the space in which the Green's function property is represented. The spaces are:

        | `space_id` | `variables` |
        | ------ | ------ |
        | 'r' | WignerSeitz |
        | 'rt' | WignerSeitz + Time |
        | 'rw' | WignerSeitz + Frequency |
        | 'rit' | WignerSeitz + ImaginaryTime |
        | 'riw' | WignerSeitz + MatsubaraFrequency |
        | 'k' | KMesh |
        | 'kt' | KMesh + Time |
        | 'kw' | KMesh + Frequency |
        | 'kit' | KMesh + ImaginaryTime |
        | 'kiw' | KMesh + MatsubaraFrequency |
        | 't' | Time |
        | 'it' | Frequency |
        | 'w' | ImaginaryTime |
        | 'iw' | MatsubaraFrequency |
        """,
    )

    def __init__(
        self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        # ! n_orbitals need to be set up during initialization of the class
        self.rank = [
            int(kwargs.get('n_atoms')),
            int(kwargs.get('n_correlated_orbitals')),
        ]

    def resolve_space_id(self) -> str:
        """
        Resolves the `space_id` based on the stored `variables` in the class.

        Returns:
            str: The resolved `space_id` of the Green's function property.
        """
        _real_space_map = {
            'r': WignerSeitz,  # ? check if this is correct
            'k': KMesh,
        }
        _time_space_map = {
            't': Time,
            'it': ImaginaryTime,
            'w': Frequency,
            'iw': MatsubaraFrequency,
        }

        def find_space_id(space_map: dict) -> str:
            """
            Finds the id string for a given map.

            Args:
                space_map (dict[str, Variables]): _description_

            Returns:
                str: _description_
            """
            for space_id, variable_cls in space_map.items():
                space_variable = get_variables(
                    variables=self.variables, variable_cls=variable_cls
                )
                if len(space_variable) > 0:
                    return space_id
            return ''

        space_id = find_space_id(_real_space_map) + find_space_id(_time_space_map)
        return space_id if space_id else None

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

        space_id = self.resolve_space_id()
        if self.space_id is not None and self.space_id != space_id:
            logger.warning(
                f'The stored `space_id`, {self.space_id}, does not coincide with the resolved one, {space_id}. We will update it.'
            )
        self.space_id = space_id


class ElectronicGreensFunction(BaseGreensFunction):
    """
    Charge-charge correlation functions.
    """

    iri = 'http://fairmat-nfdi.eu/taxonomy/ElectronicGreensFunction'

    value = Quantity(
        type=np.complex128,
        unit='1/joule',
        description="""
        Value of the electronic Green's function matrix.
        """,
    )

    def __init__(
        self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.name = self.m_def.name

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class ElectronicSelfEnergy(BaseGreensFunction):
    """
    Corrections to the energy of an electron due to its interactions with its environment.
    """

    iri = 'http://fairmat-nfdi.eu/taxonomy/ElectronicSelfEnergy'

    value = Quantity(
        type=np.complex128,
        unit='joule',
        description="""
        Value of the electronic self-energy matrix.
        """,
    )

    def __init__(
        self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.name = self.m_def.name

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class HybridizationFunction(BaseGreensFunction):
    """
    Dynamical hopping of the electrons in a lattice in and out of the reservoir or bath.
    """

    iri = 'http://fairmat-nfdi.eu/taxonomy/HybridizationFunction'

    value = Quantity(
        type=np.complex128,
        unit='joule',
        description="""
        Value of the electronic hybridization function.
        """,
    )

    def __init__(
        self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.name = self.m_def.name

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class QuasiparticleWeight(PhysicalProperty):
    """
    Renormalization of the electronic mass due to the interactions with the environment. Within the Fermi liquid
    theory of solids, this is calculated as:

        Z = 1 - ∂Σ/∂ω|ω=0

    where Σ is the `ElectronicSelfEnergy`. The quasiparticle weight is a measure of the strength of the
    electron-electron interactions and takes values between 0 and 1, with Z = 1 representing a non-correlated
    system, and Z = 0 the Mott state.
    """

    # ! we use `atoms_state_ref` and `orbitals_state_ref` to enforce order in the matrices

    iri = 'http://fairmat-nfdi.eu/taxonomy/HybridizationFunction'

    system_correlation_strengths = Quantity(
        type=MEnum(
            'non-correlated metal',
            'strongly-correlated metal',
            'OSMI',
            'Mott insulator',
        ),
        description="""
        String used to identify the type of system based on the strength of the electron-electron interactions.

        | `type` | Description |
        | ------ | ------ |
        | 'non-correlated metal' | All `value` are above 0.7. Renormalization effects are negligible. |
        | 'strongly-correlated metal' | All `value` are below 0.4 and above 0. Renormalization effects are important. |
        | 'OSMI' | Orbital-selective Mott insulator: some orbitals have a zero `value` while others a finite one. |
        | 'Mott insulator' | All `value` are 0.0. Mott insulator state. |
        """,
    )

    n_atoms = Quantity(
        type=np.int32,
        description="""
        Number of atoms involved in the correlations effect and used for the matrix representation of the quasiparticle weight.
        """,
    )

    atoms_state_ref = Quantity(
        type=AtomsState,
        shape=['n_atoms'],
        description="""
        Reference to the `AtomsState` section in which the Green's function properties are calculated.
        """,
    )

    n_correlated_orbitals = Quantity(
        type=np.int32,
        description="""
        Number of orbitals involved in the correlations effect and used for the matrix representation of the quasiparticle weight.
        """,
    )

    correlated_orbitals_ref = Quantity(
        type=OrbitalsState,
        shape=['n_correlated_orbitals'],
        description="""
        Reference to the `OrbitalsState` section in which the Green's function properties are calculated.
        """,
    )

    spin_channel = Quantity(
        type=np.int32,
        description="""
        Spin channel of the corresponding electronic property. It can take values of 0 and 1.
        """,
    )

    value = Quantity(
        type=np.float64,
        description="""
        Value of the quasiparticle weight matrices.
        """,
    )

    def __init__(
        self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        # ! n_orbitals need to be set up during initialization of the class
        self.rank = [
            int(kwargs.get('n_atoms')),
            int(kwargs.get('n_correlated_orbitals')),
        ]
        self.name = self.m_def.name

    def is_valid_quasiparticle_weight(self) -> bool:
        """
        Check if the quasiparticle weight values are valid, i.e., if all `value` are defined between
        0 and 1.

        Returns:
            (bool): True if the quasiparticle weight is valid, False otherwise.
        """
        if (self.value < 0.0).any() or (self.value > 1.0).any():
            return False
        return True

    def resolve_system_correlation_strengths(self) -> str:
        """
        Resolves the `system_correlation_strengths` of the quasiparticle weight based on the stored `value` values.

        Returns:
            str: The resolved `system_correlation_strengths` of the quasiparticle weight.
        """
        if np.all(self.value > 0.7):
            return 'non-correlated metal'
        elif np.all((self.value < 0.4) & (self.value > 0)):
            return 'strongly-correlated metal'
        elif np.any(self.value == 0) and np.any(self.value > 0):
            return 'OSMI'
        elif np.all(self.value < 1e-2):
            return 'Mott insulator'
        return None

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

        if self.is_valid_quasiparticle_weight() is False:
            logger.error(
                'Invalid negative quasiparticle weights found: could not validate them.'
            )
            return

        system_correlation_strengths = self.resolve_system_correlation_strengths()
        if (
            self.system_correlation_strengths is not None
            and self.system_correlation_strengths != system_correlation_strengths
        ):
            logger.warning(
                f'The stored `system_correlation_strengths`, {self.system_correlation_strengths}, does not coincide with the resolved one, {system_correlation_strengths}. We will update it.'
            )
        self.system_correlation_strengths = system_correlation_strengths
