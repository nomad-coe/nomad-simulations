from nomad_simulations.schema_packages.properties import TotalForce


class TestTotalForce:
    """
    Test the `TotalForce` class defined in `properties/forces.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `TotalForce` class.
        """
        total_force = TotalForce()
        # assert total_force.iri == 'http://fairmat-nfdi.eu/taxonomy/TotalForce'
        assert total_force.name == 'TotalForce'
        assert total_force.rank == []
