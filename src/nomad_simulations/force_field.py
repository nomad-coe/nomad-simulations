import numpy as np
import pint
import re
import typing
from structlog.stdlib import BoundLogger
from typing import Optional, List, Tuple
from ase.dft.kpoints import monkhorst_pack, get_monkhorst_pack_size_and_offset

from nomad.units import ureg
from nomad.datamodel.data import ArchiveSection
from nomad.datamodel.metainfo.annotations import ELNAnnotation
from nomad.metainfo import (
    Quantity,
    SubSection,
    MEnum,
    Section,
    Context,
    JSON,
)

from .model_method import ModelMethod, NumericalSettings
from .model_system import ModelSystem
from .utils import is_not_representative


class ParamEntry(ArchiveSection):
    """
    Generic section defining a parameter name and value
    """

    name = Quantity(
        type=str,
        shape=[],
        description="""
        Name of the parameter.
        """,
    )

    value = Quantity(
        type=str,
        shape=[],
        description="""
        Value of the parameter as a string.
        """,
    )

    unit = Quantity(
        type=str,
        shape=[],
        description="""
        Unit of the parameter as a string.
        """,
    )

    # TODO add description quantity


class ForceCalculations(NumericalSettings):
    """
    Section containing the parameters for force calculations according to the referenced force field
    during a molecular dynamics run.
    """

    vdw_cutoff = Quantity(
        type=np.float64,
        shape=[],
        unit="m",
        description="""
        Cutoff for calculating VDW forces.
        """,
    )

    coulomb_type = Quantity(
        type=MEnum(
            "cutoff",
            "ewald",
            "multilevel_summation",
            "particle_mesh_ewald",
            "particle_particle_particle_mesh",
            "reaction_field",
        ),
        shape=[],
        description="""
        Method used for calculating long-ranged Coulomb forces.

        Allowed values are:

        | Barostat Name          | Description                               |

        | ---------------------- | ----------------------------------------- |

        | `""`                   | No thermostat               |

        | `"Cutoff"`          | Simple cutoff scheme. |

        | `"Ewald"` | Standard Ewald summation as described in any solid-state physics text. |

        | `"Multi-Level Summation"` |  D. Hardy, J.E. Stone, and K. Schulten,
        [Parallel. Comput. **35**, 164](https://doi.org/10.1016/j.parco.2008.12.005)|

        | `"Particle-Mesh-Ewald"`        | T. Darden, D. York, and L. Pedersen,
        [J. Chem. Phys. **98**, 10089 (1993)](https://doi.org/10.1063/1.464397) |

        | `"Particle-Particle Particle-Mesh"` | See e.g. Hockney and Eastwood, Computer Simulation Using Particles,
        Adam Hilger, NY (1989). |

        | `"Reaction-Field"` | J.A. Barker and R.O. Watts,
        [Mol. Phys. **26**, 789 (1973)](https://doi.org/10.1080/00268977300102101)|
        """,
    )

    coulomb_cutoff = Quantity(
        type=np.float64,
        shape=[],
        unit="m",
        description="""
        Cutoff for calculating short-ranged Coulomb forces.
        """,
    )

    neighbor_update_frequency = Quantity(
        type=int,
        shape=[],
        description="""
        Number of timesteps between updating the neighbor list.
        """,
    )

    neighbor_update_cutoff = Quantity(
        type=np.float64,
        shape=[],
        unit="m",
        description="""
        The distance cutoff for determining the neighbor list.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

class Potential(ArchiveSection):
    """
    Section containing information about an interaction potential.
    """

    type = Quantity(
        type=str,
        shape=[],
        description="""
        Specifies the type (i.e., functional form) of the interaction potential.
        """,
    )

    parameters = Quantity(
        type=dict[str, np.float64],  # ? Used to be typing.ANY, not sure what I want here
        shape=[],
        description="""
        Dictionary of label and parameters of the interaction potential.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

class BondPotential(Potential):
    """
    Section containing information about bond potentials.

    Suggested types are: harmonic, cubic, Morse, fene, tabulated
    """

    equilibrium_value = Quantity(
        type=np.float64,
        unit='m',
        shape=[],
        description="""
        Specifies the equilibrium bond distance.
        """,
    )

    force_constant = Quantity(
        type=np.float64,
        shape=[],
        unit='J / m **2',
        description="""
        Specifies the force constant of the bond potential.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

class HarmonicBond(BondPotential):
    """
    Section containing information about a Harmonic bond potential: U(x) = 1/2 k (x-x_0)^2 + C,
    where k is the `force_constant` and x_0 is the `equilibrium_value` of x.
    C is an arbitrary constant (not stored).
    """

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)
        self.type = 'harmonic'

class CubicBond(BondPotential):
    """
    Section containing information about a Cubic bond potential: U(x) = 1/2 k (x-x_0)^2 + 1/3 k_c (x-x_0)^3,
    where k is the (harmonic) `force_constant`, k_c is the `force_constant_cubic`, and x_0 is the `equilibrium_value` of x.
    C is an arbitrary constant (not stored).
    """

    force_constant_cubic = Quantity(
        type=np.float64,
        shape=[],
        unit='J / m**3',
        description="""
        Specifies the cubic force constant of the bond potential.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)
        self.type = 'cubic'

class MorseBond(BondPotential):
    """
    Section containing information about a Morse potential: U(x) = D [1 - exp(- a (x-x_0)]^2 + C,
    where a = sqrt(k/2D) is the `well_steepness`, with `force constant` k.
    D is the `well_depth`, and x_0 is the `equilibrium_value` of x. C is an arbitrary constant (not stored).
    """

    well_depth = Quantity(
        type=np.float64,
        unit='J',
        shape=[],
        description="""
        Specifies the depth of the potential well.
        """,
    )

    well_steepness = Quantity(
        type=np.float64,
        unit='1/m',
        shape=[],
        description="""
        Specifies the steepness of the potential well.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)
        self.type = 'Morse'
        if self.well_depth is not None and self.well_steepness is not None:
            self.force_constant = 2. * self.well_depth * self.well_steepness**2
        elif self.well_depth is not None and self.force_constant is not None:
            self.well_steepness = np.sqrt(self.force_constant / (2. * self.well_depth))

class FenePotential(BondPotential):
    """
    Section containing information about a FENE potential: U(x) = -1/2 k (x_0)^2 ln[1-(x^2)/(x_0^2)] + C,
    k is the `force_constant`, and x_0 is the `equilibrium_value` of x. C is an arbitrary constant (not stored).
    """

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)
        self.type = 'fene'

class TabulatedBond(BondPotential):
    """
    Section containing information about a tabulated bond potential. The value of the potential and/or force
    is stored for a set of corresponding bin distances.
    """

    bins = Quantity(
        type=np.float64,
        unit='m',
        shape=[],
        description="""
        List of bin distances.
        """,
    )

    energies = Quantity(
        type=np.float64,
        unit='J',
        shape=[],
        description="""
        List of energy values associated with each bin.
        """,
    )

    forces = Quantity(
        type=np.float64,
        unit='J/m',
        shape=[],
        description="""
        List of force values associated with each bin.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)
        self.type = 'tabulated'
        if self.bins is not None and self.energies is not None:
            if len(self.bins != self.energies):
                logger.error('bins and energies values have different length in TabulatedBond')
        if self.bins is not None and self.forces is not None:
            if len(self.bins != self.forces):
                logger.error('bins and forces have different length in TabulatedBond')

class AnglePotential(Potential):
    """
    Section containing information about bond potentials.

    Suggested types are: harmonic, tabulated
    """

    equilibrium_value = Quantity(
        type=np.float64,
        unit='degree',
        shape=[],
        description="""
        Specifies the equilibrium angle.
        """,
    )

    force_constant = Quantity(
        type=np.float64,
        shape=[],
        unit='J / degree**2',
        description="""
        Specifies the force constant of the angle potential.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

class HarmonicAngle(AnglePotential):
    """
    Section containing information about a Harmonic angle potential: U(x) = 1/2 k (x-x_0)^2 + C,
    where k is the `force_constant` and x_0 is the `equilibrium_value` of the angle x.
    C is an arbitrary constant (not stored).
    """

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)
        self.type = 'harmonic'

class TabulatedAngle(AnglePotential):
    """
    Section containing information about a tabulated angle potential. The value of the potential and/or force
    is stored for a set of corresponding bin distances.
    """

    bins = Quantity(
        type=np.float64,
        unit='degree',
        shape=[],
        description="""
        List of bin distances.
        """,
    )

    energies = Quantity(
        type=np.float64,
        unit='J',
        shape=[],
        description="""
        List of energy values associated with each bin.
        """,
    )

    forces = Quantity(
        type=np.float64,
        unit='J/degree',
        shape=[],
        description="""
        List of force values associated with each bin.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)
        self.type = 'tabulated'
        if self.bins is not None and self.energies is not None:
            if len(self.bins != self.energies):
                logger.error('bins and energies values have different length in TabulatedBond')
        if self.bins is not None and self.forces is not None:
            if len(self.bins != self.forces):
                logger.error('bins and forces have different length in TabulatedBond')

class Interactions(ArchiveSection):
    """
    Section containing the list of particles involved in a particular type of interaction and/or any associated parameters.
    """

    type = Quantity(
        type=MEnum("bond", "angle", "dihedral", "improper dihedral"),
        shape=[],
        description="""
        Denotes the classification of the interaction.
        """,
    )

    name = Quantity(
        type=str,
        shape=[],
        description="""
        Specifies the name of the interaction. Can contain information on the species,
        cut-offs, potential versions, etc.
        """,
    )

    n_interactions = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description="""
        Total number of interactions of this interaction type-name.
        """,
    )

    n_particles = Quantity(
        type=np.int32,
        shape=[],
        description="""
        Number of particles included in (each instance of) the interaction.
        """,
    )

    particle_labels = Quantity(
        type=np.dtype(str),
        shape=["n_interactions", "n_atoms"],
        description="""
        Labels of the particles described by the interaction. In general, the structure is a list of list of tuples.
        """,
    )

    particle_indices = Quantity(
        type=np.int32,
        shape=["n_interactions", "n_atoms"],
        description="""
        Indices of the particles in the system described by the interaction. In general, the structure is a list of list of tuples.
        """,
    )

    potential = SubSection(sub_section=Potential.m_def, repeats=False)

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

class ForceField(ModelMethod):
    """
    Section containing the parameters of a (classical, particle-based) force field model.
    Typical `numerical_settings` are ForceCalculations and NeighborSearching.
    Lists of interactions by type and, if available, corresponding parameters can be given within `interactions`.
    Additionally, a published model can be referenced with `reference`.
    """

    name = Quantity(
        type=str,
        shape=[],
        description="""
        Identifies the name of the model.
        """,
    )

    reference = Quantity(
        type=str,
        shape=[],
        description="""
        Reference to the model e.g. DOI, URL.
        """,
    )

    interactions = SubSection(sub_section=Interactions.m_def, repeats=True)

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        self.name = 'ForceField'
