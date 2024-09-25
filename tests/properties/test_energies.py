from nomad_simulations.schema_packages.properties import (
    FermiLevel,
    KineticEnergy,
    PotentialEnergy,
    TotalEnergy,
)


class TestFermiLevel:
    """
    Test the `FermiLevel` class defined in `properties/energies.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `FermiLevel` class.
        """
        fermi_level = FermiLevel()
        assert fermi_level.iri == 'http://fairmat-nfdi.eu/taxonomy/FermiLevel'
        assert fermi_level.name == 'FermiLevel'
        assert fermi_level.rank == []


class TestTotalEnergy:
    """
    Test the `TotalEnergy` class defined in `properties/energies.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `TotalEnergy` class.
        """
        total_energy = TotalEnergy()
        # assert total_energy.iri == 'http://fairmat-nfdi.eu/taxonomy/TotalEnergy'
        assert total_energy.name == 'TotalEnergy'
        assert total_energy.rank == []


class TestKineticEnergy:
    """
    Test the `KineticEnergy` class defined in `properties/energies.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `KineticEnergy` class.
        """
        kinetic_energy = KineticEnergy()
        # assert kinetic_energy.iri == 'http://fairmat-nfdi.eu/taxonomy/KineticEnergy'
        assert kinetic_energy.name == 'KineticEnergy'
        assert kinetic_energy.rank == []


class TestPotentialEnergy:
    """
    Test the `PotentialEnergy` class defined in `properties/energies.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `PotentialEnergy` class.
        """
        potential_energy = PotentialEnergy()
        # assert potential_energy.iri == 'http://fairmat-nfdi.eu/taxonomy/PotentialEnergy'
        assert potential_energy.name == 'PotentialEnergy'
        assert potential_energy.rank == []
