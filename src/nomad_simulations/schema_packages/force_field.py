import numpy as np
import pint
import re
import typing
from scipy.interpolate import UnivariateSpline

# from structlog.stdlib import BoundLogger
from typing import Optional, List, Tuple
from ase.dft.kpoints import monkhorst_pack, get_monkhorst_pack_size_and_offset

from nomad.units import ureg
from nomad.datamodel.data import ArchiveSection
from nomad.datamodel.metainfo.annotations import ELNAnnotation
from nomad.metainfo import (
    URL,
    Quantity,
    SubSection,
    MEnum,
    Section,
    Context,
    JSON,
)

from nomad_simulations.schema_packages.model_system import ModelSystem
from nomad_simulations.schema_packages.model_method import BaseModelMethod, ModelMethod
from nomad_simulations.schema_packages.utils import is_not_representative

MOL = 6.022140857e23


class ParameterEntry(ArchiveSection):
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


#     # TODO add description quantity


class Potential(BaseModelMethod):
    """
    Section containing information about an interaction potential.

        name: str - potential name, can be as specific as needed
        type: str - potential type, e.g., 'bond', 'angle', 'dihedral', 'improper dihedral', 'nonbonded'
        functional_form: str - functional form of the potential, e.g., 'harmonic', 'Morse', 'Lennard-Jones'
        external_reference: URL
    """

    parameters = SubSection(
        sub_section=ParameterEntry.m_def,
        repeats=True,
        description="""
        List of parameters for custom potentials.
        """,
    )

    type = Quantity(
        type=MEnum('bond', 'angle', 'dihedral', 'improper dihedral', 'nonbonded'),
        shape=[],
        description="""
        Denotes the classification of the interaction.
        """,
    )

    functional_form = Quantity(
        type=str,
        shape=[],
        description="""
        Specifies the functional form of the interaction potential, e.g., harmonic, Morse, Lennard-Jones, etc.
        """,
    )

    n_interactions = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description="""
        Total number of interactions in the system for this potential.
        """,
    )

    n_particles = Quantity(
        type=np.int32,
        shape=[],
        description="""
        Number of particles interacting via (each instance of) this potential.
        """,
    )

    particle_labels = Quantity(
        type=np.dtype(str),
        shape=['n_interactions', 'n_particles'],
        description="""
        Labels of the particles for each instance of this potential, stored as a list of tuples.
        """,
    )

    particle_indices = Quantity(
        type=np.int32,
        shape=['n_interactions', 'n_particles'],
        description="""
        Indices of the particles for each instance of this potential, stored as a list of tuples.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        # set the dimensions based on the particle indices, if stored
        if not self.n_interactions:
            self.n_interactions = (
                len(self.particle_indices)
                if self.particle_indices is not None
                else None
            )
        if not self.n_particles:
            self.n_interactions = (
                len(self.particle_indices[0])
                if self.particle_indices is not None
                else None
            )

        # check the consistency of the dimensions of the particle indices and labels
        if self.n_interactions and self.n_particles:
            if self.particle_indices is not None:
                assert len(self.particle_indices) == self.n_interactions
                assert len(self.particle_indices[0]) == self.n_particles
            if self.particle_labels is not None:
                assert len(self.particle_labels) == self.n_interactions
                assert len(self.particle_labels[0]) == self.n_particles


class TabulatedPotential(Potential):
    """
    Abstract class for tabulated potentials. The value of the potential and/or force
    is stored for a set of corresponding bin distances. The units for bins and forces
    should be set in the individual subclasses.
    """

    bins = Quantity(
        type=np.float64,
        shape=[],
        description="""
        List of bin angles.
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
        shape=[],
        description="""
        List of force values associated with each bin.
        """,
    )

    def compute_forces(self, bins, energies, smoothing_factor=None):
        if smoothing_factor is None:
            smoothing_factor = len(bins) - np.sqrt(2 * len(bins))

        spline = UnivariateSpline(bins, energies, s=smoothing_factor)
        forces = -1.0 * spline.derivative()(bins)

        return forces

    def compute_energies(self, bins, forces, smoothing_factor=None):
        if smoothing_factor is None:
            smoothing_factor = len(bins) - np.sqrt(2 * len(bins))

        spline = UnivariateSpline(bins, forces, s=smoothing_factor)
        energies = -1.0 * np.array([spline.integral(bins[0], x) for x in bins])
        energies -= np.min(energies)

        return energies

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        self.name = 'TabulatedPotential'
        if not self.functional_form:
            self.functional_form = 'tabulated'
        elif self.functional_form != 'tabulated':
            logger.warning(f'Incorrect functional form set for {self.name}.')

        if self.bins is not None and self.energies is not None:
            if len(self.bins) != len(self.energies):
                logger.error(
                    f'bins and energies values have different length in {self.name}'
                )
        if self.bins is not None and self.forces is not None:
            if len(self.bins) != len(self.forces):
                logger.error(f'bins and forces have different length in {self.name}')

        if self.bins is not None:
            smoothing_factor = len(self.bins) - np.sqrt(2 * len(self.bins))
            tol = 1e-2
            if self.forces is None and self.energies is not None:
                try:
                    # generate forces from energies numerically using spline
                    self.forces = (
                        self.compute_forces(
                            self.bins.magnitude,
                            self.energies.magnitude,
                            smoothing_factor=smoothing_factor,
                        )  # ? How do I deal with units here?
                    )
                    # re-derive energies to check consistency of the forces
                    energies = (
                        self.compute_energies(
                            self.bins.magnitude,
                            self.forces.magnitude,
                            smoothing_factor=smoothing_factor,
                        )
                        * ureg.J
                    )

                    energies_diff = energies.to('kJ').magnitude * MOL - (
                        self.energies.to('kJ').magnitude * MOL
                        - np.min(self.energies.to('kJ').magnitude * MOL)
                    )
                    if np.all([x < tol for x in energies_diff]):
                        logger.warning(
                            f'Tabulated forces were generated from energies in {self.name},'
                            f'with consistency errors less than tol={tol}. '
                        )
                    else:
                        logger.warning(
                            f'Unable to derive tabulated forces from energies in {self.name},'
                            f'consistency errors were greater than tol={tol}.'
                        )
                        self.forces = None
                except ValueError as e:
                    logger.warning(
                        f'Unable to derive tabulated forces from energies in {self.name},'
                        f'Unkown error occurred in derivation: {e}'
                    )

            if self.forces is not None and self.energies is None:
                print('in gen energies')
                try:
                    # generated energies from forces numerically using spline
                    self.energies = self.compute_energies(
                        self.bins.magnitude,
                        self.forces.magnitude,
                        smoothing_factor=smoothing_factor,
                    )
                    # re-derive forces to check consistency of the energies
                    forces = (
                        self.compute_forces(
                            self.bins.magnitude,
                            self.energies.magnitude,
                            smoothing_factor=smoothing_factor,
                        )
                        * ureg.J
                        / self.bins.units
                    )

                    forces_diff = forces.to(f'kJ/{self.bins.units}').magnitude * MOL - (
                        self.forces.to(f'kJ/{self.bins.units}').magnitude * MOL
                    )
                    if np.all([x < tol for x in forces_diff]):
                        logger.warning(
                            f'Tabulated energies were generated from forces in {self.name},'
                            f'with consistency errors less than tol={tol}. '
                        )
                    else:
                        logger.warning(
                            f'Unable to derive tabulated energies from forces in {self.name},'
                            f'consistency errors were greater than tol={tol}.'
                        )
                        self.energies = None
                except ValueError as e:
                    logger.warning(
                        f'Unable to derive tabulated energies from forces in {self.name},'
                        f'Unkown error occurred in derivation: {e}'
                    )


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

        if not self.name:
            self.name = 'BondPotential'
        if not self.type:
            self.type = 'bond'
        elif self.type != 'bond':
            logger.warning('Incorrect type set for BondPotential.')

        if self.n_particles:
            if self.n_particles != 2:
                logger.warning('Incorrect number of particles set for BondPotential.')
            else:
                self.n_particles = 2


class HarmonicBond(BondPotential):
    """
    Section containing information about a Harmonic bond potential: U(x) = 1/2 k (x-x_0)^2 + C,
    where k is the `force_constant` and x_0 is the `equilibrium_value` of x.
    C is an arbitrary constant (not stored).
    """

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        self.name = 'HarmonicBond'
        if not self.functional_form:
            self.functional_form = 'harmonic'
        elif self.functional_form != 'harmonic':
            logger.warning('Incorrect functional form set for HarmonicBond.')


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

        self.name = 'CubicBond'
        if not self.functional_form:
            self.functional_form = 'cubic'
        elif self.functional_form != 'cubic':
            logger.warning('Incorrect functional form set for CubicBond.')


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

        self.name = 'MorseBond'
        if not self.functional_form:
            self.functional_form = 'morse'
        elif self.functional_form != 'morse':
            logger.warning('Incorrect functional form set for MorseBond.')

        if self.well_depth is not None and self.well_steepness is not None:
            self.force_constant = 2.0 * self.well_depth * self.well_steepness**2
        elif self.well_depth is not None and self.force_constant is not None:
            self.well_steepness = np.sqrt(self.force_constant / (2.0 * self.well_depth))


class FeneBond(BondPotential):
    """
    Section containing information about a FENE potential: U(x) = -1/2 k (X_0)^2 ln[1-((x-x_0)^2)/(X_0^2)] + C,
    k is the `force_constant`, x_0 is the `equilibrium_value` of x, and X_0 is the maximum allowable bond extension beyond x_0. C is an arbitrary constant (not stored).
    """

    maximum_extension = Quantity(
        type=np.float64,
        unit='m',
        shape=[],
        description="""
        Specifies the maximum extension beyond the equilibrium bond distance.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        self.name = 'FeneBond'
        if not self.functional_form:
            self.functional_form = 'fene'
        elif self.functional_form != 'fene':
            logger.warning('Incorrect functional form set for FeneBond.')


class TabulatedBond(TabulatedPotential, BondPotential):
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

        self.name = 'TabulatedBond'


class AnglePotential(Potential):
    """
    Section containing information about angle potentials.

    Suggested types are: ... ? harmonic, cubic, Morse, fene, tabulated
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

        self.name = 'AnglePotential'
        if not self.type:
            self.type = 'angle'
        elif self.type != 'angle':
            logger.warning('Incorrect type set for AnglePotential.')

        if self.n_particles:
            if self.n_particles != 3:
                logger.warning('Incorrect number of particles set for AnglePotential.')
            else:
                self.n_particles = 3


class HarmonicAngle(AnglePotential):
    """
    Section containing information about a Harmonic angle potential: U(x) = 1/2 k (x-x_0)^2 + C,
    where k is the `force_constant` and x_0 is the `equilibrium_value` of the angle x.
    C is an arbitrary constant (not stored).
    """

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        self.name = 'HarmonicAngle'
        if not self.functional_form:
            self.functional_form = 'harmonic'
        elif self.functional_form != 'harmonic':
            logger.warning('Incorrect functional form set for HarmonicAngle.')


class TabulatedAngle(AnglePotential, TabulatedPotential):
    """
    Section containing information about a tabulated bond potential. The value of the potential and/or force
    is stored for a set of corresponding bin distances.
    """

    bins = Quantity(
        type=np.float64,
        unit='degree',
        shape=[],
        description="""
        List of bin angles.
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

        self.name = 'TabulatedAngle'


# class Interactions(ArchiveSection):
#     """
#     Section containing the list of particles involved in a particular type of interaction and/or any associated parameters.
#     """

#     type = Quantity(
#         type=MEnum('bond', 'angle', 'dihedral', 'improper dihedral'),
#         shape=[],
#         description="""
#         Denotes the classification of the interaction.
#         """,
#     )

#     name = Quantity(
#         type=str,
#         shape=[],
#         description="""
#         Specifies the name of the interaction. Can contain information on the species,
#         cut-offs, potential versions, etc.
#         """,
#     )

#     n_interactions = Quantity(
#         type=np.dtype(np.int32),
#         shape=[],
#         description="""
#         Total number of interactions of this interaction type-name.
#         """,
#     )

#     n_particles = Quantity(
#         type=np.int32,
#         shape=[],
#         description="""
#         Number of particles included in (each instance of) the interaction.
#         """,
#     )

#     particle_labels = Quantity(
#         type=np.dtype(str),
#         shape=['n_interactions', 'n_atoms'],
#         description="""
#         Labels of the particles described by the interaction. In general, the structure is a list of list of tuples.
#         """,
#     )

#     particle_indices = Quantity(
#         type=np.int32,
#         shape=['n_interactions', 'n_atoms'],
#         description="""
#         Indices of the particles in the system described by the interaction. In general, the structure is a list of list of tuples.
#         """,
#     )

#     potential = SubSection(sub_section=Potential.m_def, repeats=False)

#     def normalize(self, archive, logger) -> None:
#         super().normalize(archive, logger)


class ForceField(ModelMethod):
    """
    Section containing the parameters of a (classical, particle-based) force field model.
    Typical `numerical_settings` are ForceCalculations.
    Lists of interactions by type and, if available, corresponding parameters can be given within `interactions`.
    Additionally, a published model can be referenced with `reference`.
    """

    # name and external reference already defined in BaseModelMethod
    # name = Quantity(
    #     type=str,
    #     shape=[],
    #     description="""
    #     Identifies the name of the model.
    #     """,
    # )

    # reference = Quantity(
    #     type=str,
    #     shape=['0..*'],
    #     description="""
    #     List of references to the model e.g. DOI, URL.
    #     """,
    # )

    kimid = Quantity(
        type=URL,
        description="""
        Reference to a model stored on the OpenKim database.
        """,
        a_eln=ELNAnnotation(component='URLEditQuantity'),
    )

    #     interactions = SubSection(sub_section=Interactions.m_def, repeats=True)
    contributions = SubSection(
        sub_section=Potential.m_def,
        repeats=True,
        description="""
        Contribution or sub-term of the total model Hamiltonian.
        """,
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        self.name = 'ForceField'
        logger.warning('in force field')
