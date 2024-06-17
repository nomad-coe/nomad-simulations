from nomad.config.models.plugins import SchemaPackageEntryPoint
from pydantic import Field


class NOMADSimulationsEntryPoint(SchemaPackageEntryPoint):
    parameter: int = Field(0, description='Custom configuration parameter')

    def load(self):
        from nomad_simulations.schema.general import m_package

        return m_package


nomad_simulations_plugin = NOMADSimulationsEntryPoint(
    name='NOMADSimulations',
    description='A NOMAD plugin for FAIR schemas for simulation data.',
)
