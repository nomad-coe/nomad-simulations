from nomad.config.models.plugins import SchemaPackageEntryPoint
from pydantic import Field


class NOMADSimulationsEntryPoint(SchemaPackageEntryPoint):
    dos_energy_tolerance: float = Field(
        8.01088e-21,
        description='Tolerance (in joules) of the DOS energies to match the reference of energies in the DOS normalize function.',
    )
    dos_intensities_threshold: float = Field(
        1e-8,
        description='Threshold value (in joules^-1) at which the DOS intensities are considered non-zero.',
    )
    occupation_tolerance: float = Field(
        1e-3,
        description='Tolerance for the occupation of a eigenstate to be non-occupied.',
    )
    fermi_surface_tolerance: float = Field(
        1e-8,
        description='Tolerance (in joules) for energies to be close to the Fermi level and hence define the Fermi surface of a material.',
    )
    symmetry_tolerance: float = Field(
        0.1, description='Tolerance for the symmetry analyzer used from MatID.'
    )
    cluster_threshold: float = Field(
        2.5,
        description='Threshold for the distance between atoms to be considered in the same cluster.',
    )
    limit_system_type_classification: float = Field(
        64,
        description='Limite of the number of atoms in the unit cell to be treated for the system type classification from MatID to work. This is done to avoid overhead of the package.',
    )
    equal_cell_positions_tolerance: float = Field(
        12,
        description='Decimal order or tolerance (in meters) for comparing cell positions.',
    )

    def load(self):
        from nomad_simulations.schema_packages.general import m_package

        return m_package


nomad_simulations_plugin = NOMADSimulationsEntryPoint(
    name='NOMADSimulations',
    description='A NOMAD plugin for FAIR schemas for simulation data.',
)
