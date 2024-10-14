import numbers
from typing import TYPE_CHECKING, Any, Optional, Union

import ase
import ase.geometry
import numpy as np
import pint
from deprecated import deprecated
from nomad.datamodel.data import ArchiveSection
from nomad.datamodel.metainfo.annotations import ELNAnnotation
from nomad.datamodel.metainfo.basesections import Entity
from nomad.metainfo import MEnum, Quantity, SubSection
from nomad.units import ureg

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from nomad.metainfo import Context, Section
    from structlog.stdlib import BoundLogger


class Particles:
    """Particle object.

    Adaptation of the ASE Atoms object to coarse-grained particles. For use with
    nomad_simulations.model_system.ParticlesCell.
    Implemented methods: set_pbc,       get_pbc,
                         set_cell,      get_cell,
                         set_positions, get_positions,
                                        get_particle_types

    Parameters:

    types: str (formula) or list of str.
        Default: [‘A’]
    typeid: int
        (Optional) Id numbers of corresponding particle types.
        Default: 0
    type_shapes: str
        Store a per-type shape definition for visualization.
        A dictionary is stored for each of the NT types, corresponding
        to a shape for visualization of that type.
        Default: empty
    masses: list of float
        The mass of each particle.
        Default: 1.0
    charges: list of float
        Initial atomic charges.
        Default: 0.0
    diameter: float
        The diameter of each particle.
        Default: 1.0
    body: int
        The composite body associated with each particle. The value -1
        indicates no body.
        Default: -1
    moment_inertia: float
        The moment_inertia of each particle (I_xx, I_yy, I_zz).
        This inertia tensor is diagonal in the body frame of the particle.
        The default value is for point particles.
        Default: 0, 0, 0
    positions: float, list of xyz-positions
        Particle positions. Needs to be convertible to an
        ndarray of shape (N, 3).
        Default: 0, 0, 0
    scaled_positions: list of scaled-positions
        Like positions, but given in units of the unit cell.
        Can not be set at the same time as positions.
        Default: 0, 0, 0
    orientation: float
        The orientation of each particle. In scalar + vector notation,
        this is (r, a_x, a_y, a_z), where the quaternion is q = r + a_xi + a_yj + a_zk.
        A unit quaternion has the property: sqrt(r^2 + a_x^2 + a_y^2 + a_z^2) = 1.
        Default: 0, 0, 0, 0
    angmom: float
        The angular momentum of each particle as a quaternion.
        Default: 0, 0, 0, 0
    image: int
        The number of times each particle has wrapped around the box (i_x, i_y, i_z).
        Default: 0, 0, 0
    cell: 3x3 matrix or length 3 or 6 vector
        Unit cell vectors.  Can also be given as just three
        numbers for orthorhombic cells, or 6 numbers, where
        first three are lengths of unit cell vectors, and the
        other three are angles between them (in degrees), in following order:
        [len(a), len(b), len(c), angle(b,c), angle(a,c), angle(a,b)].
        First vector will lie in x-direction, second in xy-plane,
        and the third one in z-positive subspace.
        Default value: [0, 0, 0].
    celldisp: Vector
        Unit cell displacement vector. To visualize a displaced cell
        around the center of mass of a Systems of atoms. Default value
        = (0,0,0)
    pbc: one or three bool
        Periodic boundary conditions flags.  Examples: True,
        False, 0, 1, (1, 1, 0), (True, False, False).  Default
        value: False.

    Examples:

    These three are equivalent:

    >>> d = 1.104  # N2 bondlength
    >>> a = Atoms('N2', [(0, 0, 0), (0, 0, d)])
    >>> a = Atoms(numbers=[7, 7], positions=[(0, 0, 0), (0, 0, d)])
    >>> a = Atoms([Atom('N', (0, 0, 0)), Atom('N', (0, 0, d))])

    FCC gold:

    >>> a = 4.05  # Gold lattice constant
    >>> b = a / 2
    >>> fcc = Atoms('Au',
    ...             cell=[(0, b, b), (b, 0, b), (b, b, 0)],
    ...             pbc=True)

    Hydrogen wire:

    >>> d = 0.9  # H-H distance
    >>> h = Atoms('H', positions=[(0, 0, 0)],
    ...           cell=(d, 0, 0),
    ...           pbc=(1, 0, 0))
    """

    def __init__(
        self,
        types=None,
        positions=None,
        typeid=None,
        type_shapes=None,
        moment_inertia=None,
        masses=None,
        angmom=None,
        charges=None,
        diameter=None,
        body=None,
        scaled_positions=None,
        orientation=None,
        image=None,
        cell=None,
        pbc=None,
        celldisp=None,
    ):
        self._cellobj = self.set_cell()
        self._pbc = np.zeros(3, bool)

        particles = None

        #     if hasattr(types, 'get_positions'):
        #         atoms = types
        #         types = None
        #     elif (
        #         isinstance(types, (list, tuple))
        #         and len(types) > 0
        #         and isinstance(types[0], Atom)
        #     ):
        #         # Get data from a list or tuple of Atom objects:
        #         data = [
        #             [atom.get_raw(name) for atom in types]
        #             for name in [
        #                 'position',
        #                 'number',
        #                 'tag',
        #                 'momentum',
        #                 'mass',
        #                 'magmom',
        #                 'charge',
        #             ]
        #         ]
        #         atoms = self.__class__(None, *data)
        #         types = None

        #     if atoms is not None:
        #         # Get data from another Atoms object:
        #         if scaled_positions is not None:
        #             raise NotImplementedError
        #         if types is None and numbers is None:
        #             numbers = atoms.get_atomic_numbers()
        #         if positions is None:
        #             positions = atoms.get_positions()
        #         if tags is None and atoms.has('tags'):
        #             tags = atoms.get_tags()
        #         if momenta is None and atoms.has('momenta'):
        #             momenta = atoms.get_momenta()
        #         if magmoms is None and atoms.has('initial_magmoms'):
        #             magmoms = atoms.get_initial_magnetic_moments()
        #         if masses is None and atoms.has('masses'):
        #             masses = atoms.get_masses()
        #         if charges is None and atoms.has('initial_charges'):
        #             charges = atoms.get_initial_charges()
        #         if cell is None:
        #             cell = atoms.get_cell()
        #         if celldisp is None:
        #             celldisp = atoms.get_celldisp()
        #         if pbc is None:
        #             pbc = atoms.get_pbc()

        #     self.arrays = {}

        #     if types is None:
        #         if numbers is None:
        #             if positions is not None:
        #                 natoms = len(positions)
        #             elif scaled_positions is not None:
        #                 natoms = len(scaled_positions)
        #             else:
        #                 natoms = 0
        #             numbers = np.zeros(natoms, int)
        #         self.new_array('numbers', numbers, int)
        #     else:
        #         if numbers is not None:
        #             raise TypeError('Use only one of "types" and "numbers".')
        #         else:
        #             self.new_array('numbers', types2numbers(types), int)

        #     if self.numbers.ndim != 1:
        #         raise ValueError('"numbers" must be 1-dimensional.')

        if cell is None:
            cell = np.zeros((3, 3))
        self.set_cell(cell)

    #     if celldisp is None:
    #         celldisp = np.zeros(shape=(3, 1))
    #     self.set_celldisp(celldisp)

    #     if positions is None:
    #         if scaled_positions is None:
    #             positions = np.zeros((len(self.arrays['numbers']), 3))
    #         else:
    #             assert self.cell.rank == 3
    #             positions = np.dot(scaled_positions, self.cell)
    #     else:
    #         if scaled_positions is not None:
    #             raise TypeError('Use only one of "types" and "numbers".')
    #     self.new_array('positions', positions, float, (3,))
    #     self.set_tags(default(tags, 0))
    #     self.set_masses(default(masses, None))
    #     self.set_initial_magnetic_moments(default(magmoms, 0.0))
    #     self.set_initial_charges(default(charges, 0.0))
    #     if pbc is None:
    #         pbc = False
    #     self.set_pbc(pbc)
    #     self.set_momenta(default(momenta, (0.0, 0.0, 0.0)), apply_constraint=False)

    #     if velocities is not None:
    #         if momenta is None:
    #             self.set_velocities(velocities)
    #         else:
    #             raise TypeError('Use only one of "momenta" and "velocities".')

    #     if info is None:
    #         self.info = {}
    #     else:
    #         self.info = dict(info)

    #     self.calc = calculator

    # def set_cell(self, cell):
    #     if cell is None:
    #         cell = np.zeros((3, 3))

    # @property
    # def symbols(self):
    #     """Get chemical symbols as a :class:`ase.symbols.Symbols` object.

    #     The object works like ``atoms.numbers`` except its values
    #     are strings.  It supports in-place editing."""
    #     return Symbols(self.numbers)

    # @symbols.setter
    # def symbols(self, obj):
    #     new_symbols = Symbols.fromsymbols(obj)
    #     self.numbers[:] = new_symbols.numbers

    def get_particle_types(self):
        """Get list of particle type strings.

        Labels describing type of coarse-grained particles."""
        return list(self.types)

    def set_cell(self, cell, scale_atoms=False, apply_constraint=True):
        """Set unit cell vectors.

        Parameters:

        cell: 3x3 matrix or length 3 or 6 vector
            Unit cell.  A 3x3 matrix (the three unit cell vectors) or
            just three numbers for an orthorhombic cell. Another option is
            6 numbers, which describes unit cell with lengths of unit cell
            vectors and with angles between them (in degrees), in following
            order: [len(a), len(b), len(c), angle(b,c), angle(a,c),
            angle(a,b)].  First vector will lie in x-direction, second in
            xy-plane, and the third one in z-positive subspace.
        scale_atoms: bool
            Fix atomic positions or move atoms with the unit cell?
            Default behavior is to *not* move the atoms (scale_atoms=False).
        apply_constraint: bool
            Whether to apply constraints to the given cell.

        Examples:

        Two equivalent ways to define an orthorhombic cell:

        >>> atoms = Atoms('He')
        >>> a, b, c = 7, 7.5, 8
        >>> atoms.set_cell([a, b, c])
        >>> atoms.set_cell([(a, 0, 0), (0, b, 0), (0, 0, c)])

        FCC unit cell:

        >>> atoms.set_cell([(0, b, b), (b, 0, b), (b, b, 0)])

        Hexagonal unit cell:

        >>> atoms.set_cell([a, a, c, 90, 90, 120])

        Rhombohedral unit cell:

        >>> alpha = 77
        >>> atoms.set_cell([a, a, a, alpha, alpha, alpha])
        """

        # Override pbcs if and only if given a Cell object:
        cell = ase.Cell.new(cell)

        # XXX not working well during initialize due to missing _constraints
        if apply_constraint and hasattr(self, '_constraints'):
            for constraint in self.constraints:
                if hasattr(constraint, 'adjust_cell'):
                    constraint.adjust_cell(self, cell)

        if scale_atoms:
            M = np.linalg.solve(self.cell.complete(), cell.complete())
            self.positions[:] = np.dot(self.positions, M)

        self.cell[:] = cell

    def get_cell(self, complete=False):
        """Get the three unit cell vectors as a `class`:ase.cell.Cell` object.

        The Cell object resembles a 3x3 ndarray, and cell[i, j]
        is the jth Cartesian coordinate of the ith cell vector."""
        if complete:
            cell = self.cell.complete()
        else:
            cell = self.cell.copy()

        return cell

    @property
    def pbc(self):
        """Reference to pbc-flags for in-place manipulations."""
        return self._pbc

    @pbc.setter
    def pbc(self, pbc):
        self._pbc[:] = pbc

    def set_pbc(self, pbc):
        """Set periodic boundary condition flags."""
        self.pbc = pbc

    def get_pbc(self):
        """Get periodic boundary condition flags."""
        return self.pbc.copy()

    def set_positions(self, newpositions, apply_constraint=True):
        """Set positions, honoring any constraints. To ignore constraints,
        use *apply_constraint=False*."""
        if self.constraints and apply_constraint:
            newpositions = np.array(newpositions, float)
            for constraint in self.constraints:
                constraint.adjust_positions(self, newpositions)

        self.set_array('positions', newpositions, shape=(3,))

    def get_positions(self, wrap=False, **wrap_kw):
        """Get array of positions.

        Parameters:

        wrap: bool
            wrap atoms back to the cell before returning positions
        wrap_kw: (keyword=value) pairs
            optional keywords `pbc`, `center`, `pretty_translation`, `eps`,
            see :func:`ase.geometry.wrap_positions`
        """
        if wrap:
            if 'pbc' not in wrap_kw:
                wrap_kw['pbc'] = self.pbc
            return ase.geometry.wrap_positions(self.positions, self.cell, **wrap_kw)
        else:
            return self.arrays['positions'].copy()

    def get_scaled_positions(self, wrap=True):
        """Get positions relative to unit cell.

        If wrap is True, atoms outside the unit cell will be wrapped into
        the cell in those directions with periodic boundary conditions
        so that the scaled coordinates are between zero and one.

        If any cell vectors are zero, the corresponding coordinates
        are evaluated as if the cell were completed using
        ``cell.complete()``.  This means coordinates will be Cartesian
        as long as the non-zero cell vectors span a Cartesian axis or
        plane."""

        fractional = self.cell.scaled_positions(self.positions)

        if wrap:
            for i, periodic in enumerate(self.pbc):
                if periodic:
                    # Yes, we need to do it twice.
                    # See the scaled_positions.py test.
                    fractional[:, i] %= 1.0
                    fractional[:, i] %= 1.0

        return fractional

    def set_scaled_positions(self, scaled):
        """Set positions relative to unit cell."""
        self.positions[:] = self.cell.cartesian_positions(scaled)

    def wrap(self, **wrap_kw):
        """Wrap positions to unit cell.

        Parameters:

        wrap_kw: (keyword=value) pairs
            optional keywords `pbc`, `center`, `pretty_translation`, `eps`,
            see :func:`ase.geometry.wrap_positions`
        """

        if 'pbc' not in wrap_kw:
            wrap_kw['pbc'] = self.pbc

        self.positions[:] = self.get_positions(wrap=True, **wrap_kw)

    def _get_positions(self):
        """Return reference to positions-array for in-place manipulations."""
        return self.arrays['positions']

    def _set_positions(self, pos):
        """Set positions directly, bypassing constraints."""
        self.arrays['positions'][:] = pos

    positions = property(
        _get_positions,
        _set_positions,
        doc='Attribute for direct ' + 'manipulation of the positions.',
    )


# ? How generic (usable for any CG model) vs. Martini-specific do we want to be?
class ParticlesState(Entity):
    """
    A base section to define individual coarse-grained (CG) particle information.
    """

    # ? What do we want to qualify as type identifier? What safety checks do we need?
    particle_type = Quantity(
        type=str,
        description="""
        Symbol(s) describing the CG particle type. Currently, entrie particle label is
        used for type definition.
        """,
    )

    # ? Do we want to reflect the Martini size nomenclature and include bead volume/bead mass?
    # particle_size = Quantity(
    #     type=np.float64,
    #     description="""
    #     Particle size, determining the number of non-hydrogen atoms represented by the
    #     particle. Currently, possible values are 0.47 nm (regular, default),
    #     0.43/0.41 nm (small), and 0.34 nm (tiny).
    #     """,
    # )

    # particle_mass = Quantity(
    #     type=np.float64,
    #     description="""
    #     Particle size, determining the number of non-hydrogen atoms represented by the
    #     particle. Currently, possible values are 72 amu (regular, default), 54/45 amu
    #     (small), and 36 amu (tiny).
    #     """,
    # )

    charge = Quantity(
        type=np.int32,
        default=0,
        description="""
        Charge of the particle. Neutral = 0. Can be any positive integer (+1, +2...)
        for cations or any negative integer (-1, -2...) for anions.
        """,
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )

    def resolve_particle_type(self, logger: 'BoundLogger') -> Optional[str]:
        """
        Checks if any value is passed as particle label. Converts to string to be used as
        type identifier for the CG particle.

        Args:
            logger (BoundLogger): The logger to log messages.

        Returns:
            (Optional[str]): The resolved `particle type`.
        """
        if self.particle_type is not None and self.particle_type.isascii():
            try:
                return str(self.particle_type)
            except TypeError:
                logger.error('The parsed `particle type` can not be read.')
        return None

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

        # Get particle_type as string, if possible.
        if not isinstance(self.particle_type, str):
            self.particle_type = self.resolve_particle_type(logger=logger)
