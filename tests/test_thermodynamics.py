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

from nomad_simulations.schema_packages.properties import (
    Pressure,
    Volume,
    Temperature,
    Heat,
    Work,
    InternalEnergy,
    Enthalpy,
    Entropy,
    GibbsFreeEnergy,
    HelmholtzFreeEnergy,
    ChemicalPotential,
    HeatCapacity,
    VirialTensor,
    MassDensity,
    Hessian,
)


class TestPressure:
    """
    Test the `Pressure` class defined in `properties/thermodynamics.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `Pressure` class.
        """
        pressure = Pressure()
        assert pressure.iri == 'http://fairmat-nfdi.eu/taxonomy/Pressure'
        assert pressure.name == 'Pressure'
        assert pressure.rank == []


class TestVolume:
    """
    Test the `Volume` class defined in `properties/thermodynamics.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `Volume` class.
        """
        volume = Volume()
        assert volume.iri == 'http://fairmat-nfdi.eu/taxonomy/Volume'
        assert volume.name == 'Volume'
        assert volume.rank == []


class TestTemperature:
    """
    Test the `Temperature` class defined in `properties/thermodynamics.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `Temperature` class.
        """
        temperature = Temperature()
        assert temperature.iri == 'http://fairmat-nfdi.eu/taxonomy/Temperature'
        assert temperature.name == 'Temperature'
        assert temperature.rank == []


class TestHeat:
    """
    Test the `Heat` class defined in `properties/thermodynamics.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `Heat` class.
        """
        heat = Heat()
        assert heat.iri == 'http://fairmat-nfdi.eu/taxonomy/Heat'
        assert heat.name == 'Heat'
        assert heat.rank == []


class TestWork:
    """
    Test the `Work` class defined in `properties/thermodynamics.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `Work` class.
        """
        work = Work()
        assert work.iri == 'http://fairmat-nfdi.eu/taxonomy/Work'
        assert work.name == 'Work'
        assert work.rank == []


class TestInternalEnergy:
    """
    Test the `InternalEnergy` class defined in `properties/thermodynamics.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `InternalEnergy` class.
        """
        internal_energy = InternalEnergy()
        assert internal_energy.iri == 'http://fairmat-nfdi.eu/taxonomy/InternalEnergy'
        assert internal_energy.name == 'InternalEnergy'
        assert internal_energy.rank == []


class TestEnthalpy:
    """
    Test the `Enthalpy` class defined in `properties/thermodynamics.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `Enthalpy` class.
        """
        enthalpy = Enthalpy()
        assert enthalpy.iri == 'http://fairmat-nfdi.eu/taxonomy/Enthalpy'
        assert enthalpy.name == 'Enthalpy'
        assert enthalpy.rank == []


class TestEntropy:
    """
    Test the `Entropy` class defined in `properties/thermodynamics.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `Entropy` class.
        """
        entropy = Entropy()
        assert entropy.iri == 'http://fairmat-nfdi.eu/taxonomy/Entropy'
        assert entropy.name == 'Entropy'
        assert entropy.rank == []


class TestGibbsFreeEnergy:
    """
    Test the `GibbsFreeEnergy` class defined in `properties/thermodynamics.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `GibbsFreeEnergy` class.
        """
        gibbs_free_energy = GibbsFreeEnergy()
        assert (
            gibbs_free_energy.iri == 'http://fairmat-nfdi.eu/taxonomy/GibbsFreeEnergy'
        )
        assert gibbs_free_energy.name == 'GibbsFreeEnergy'
        assert gibbs_free_energy.rank == []


class TestHelmholtzFreeEnergy:
    """
    Test the `HelmholtzFreeEnergy` class defined in `properties/thermodynamics.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `HelmholtzFreeEnergy` class.
        """
        helmholtz_free_energy = HelmholtzFreeEnergy()
        assert (
            helmholtz_free_energy.iri
            == 'http://fairmat-nfdi.eu/taxonomy/HelmholtzFreeEnergy'
        )
        assert helmholtz_free_energy.name == 'HelmholtzFreeEnergy'
        assert helmholtz_free_energy.rank == []


class TestChemicalPotential:
    """
    Test the `ChemicalPotential` class defined in `properties/thermodynamics.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `ChemicalPotential` class.
        """
        chemical_potential = ChemicalPotential()
        assert (
            chemical_potential.iri
            == 'http://fairmat-nfdi.eu/taxonomy/ChemicalPotential'
        )
        assert chemical_potential.name == 'ChemicalPotential'
        assert chemical_potential.rank == []


class TestHeatCapacity:
    """
    Test the `HeatCapacity` class defined in `properties/thermodynamics.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `HeatCapacity` class.
        """
        heat_capacity = HeatCapacity()
        assert heat_capacity.iri == 'http://fairmat-nfdi.eu/taxonomy/HeatCapacity'
        assert heat_capacity.name == 'HeatCapacity'
        assert heat_capacity.rank == []


class TestVirialTensor:
    """
    Test the `VirialTensor` class defined in `properties/thermodynamics.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `VirialTensor` class.
        """
        virial_tensor = VirialTensor()
        assert virial_tensor.iri == 'http://fairmat-nfdi.eu/taxonomy/VirialTensor'
        assert virial_tensor.name == 'VirialTensor'
        assert virial_tensor.rank == []


class TestMassDensity:
    """
    Test the `MassDensity` class defined in `properties/thermodynamics.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `MassDensity` class.
        """
        mass_density = MassDensity()
        assert mass_density.iri == 'http://fairmat-nfdi.eu/taxonomy/MassDensity'
        assert mass_density.name == 'MassDensity'
        assert mass_density.rank == []


class TestHessian:
    """
    Test the `Hessian` class defined in `properties/thermodynamics.py`.
    """

    # ! Include this initial `test_default_quantities` method when testing your PhysicalProperty classes
    def test_default_quantities(self):
        """
        Test the default quantities assigned when creating an instance of the `Hessian` class.
        """
        hessian = Hessian()
        assert hessian.iri == 'http://fairmat-nfdi.eu/taxonomy/Hessian'
        assert hessian.name == 'Hessian'
        assert hessian.rank == []
