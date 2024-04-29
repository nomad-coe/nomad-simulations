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

import numpy as np
import pytest

from nomad.units import ureg
from nomad.datamodel import EntryArchive
from nomad.datamodel.data import ArchiveSection
from nomad.metainfo import Quantity, Section, Context, SubSection

from . import logger

from nomad_simulations.variables import Variables
from nomad_simulations.physical_property import PhysicalProperty


class DummyPhysicalProperty(PhysicalProperty):
    m_def = Section(
        iri='http://fairmat-nfdi.eu/taxonomy/DummyPhysicalProperty', rank=[3, 3]
    )

    value = Quantity(
        type=np.float64,
        unit='eV',
        description="""
        This value is defined in order to test the `__setattr__` method in `PhysicalProperty`.
        """,
    )

    def __init__(
        self, m_def: Section = None, m_context: Context = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)


class SectionWithProperties(ArchiveSection):
    m_def = Section()

    physical_property = SubSection(sub_section=PhysicalProperty.m_def, repeats=True)


class TestPhysicalProperty:
    """
    Test the `PhysicalProperty` class defined in `physical_property.py`.
    """

    @pytest.mark.parametrize(
        'rank, variables, result_variables_shape, result_full_shape',
        [
            ([], [], [], []),
            ([3], [], [], [3]),
            ([3, 3], [], [], [3, 3]),
            ([], [Variables(n_grid_points=4)], [4], [4]),
            ([3], [Variables(n_grid_points=4)], [4], [4, 3]),
            ([3, 3], [Variables(n_grid_points=4)], [4], [4, 3, 3]),
            (
                [],
                [Variables(n_grid_points=4), Variables(n_grid_points=10)],
                [4, 10],
                [4, 10],
            ),
            (
                [3],
                [Variables(n_grid_points=4), Variables(n_grid_points=10)],
                [4, 10],
                [4, 10, 3],
            ),
            (
                [3, 3],
                [Variables(n_grid_points=4), Variables(n_grid_points=10)],
                [4, 10],
                [4, 10, 3, 3],
            ),
        ],
    )
    def test_static_properties(
        self,
        rank: list,
        variables: list,
        result_variables_shape: list,
        result_full_shape: list,
    ):
        """
        Test the static properties of the `PhysicalProperty` class, `variables_shape` and `full_shape`.
        """
        physical_property = PhysicalProperty(
            m_def=Section(
                iri='http://fairmat-nfdi.eu/taxonomy/PhysicalProperty', rank=rank
            )
        )
        physical_property.source = 'simulation'
        physical_property.variables = variables
        assert physical_property.variables_shape == result_variables_shape
        assert physical_property.full_shape == result_full_shape

    def test_setattr_value(self):
        """
        Test the `__setattr__` method when setting the `value` quantity of a physical property.
        """
        physical_property = DummyPhysicalProperty(
            source='simulation',
            variables=[
                Variables(n_grid_points=4),
                Variables(n_grid_points=10),
            ],
        )
        # `physical_property.value` must have full_shape=[4, 10, 3, 3]
        value = np.ones((4, 10, 3, 3)) * ureg.eV
        assert physical_property.full_shape == list(value.shape)
        physical_property.value = value
        assert np.all(physical_property.value == value)

    def test_setattr_value_wrong_shape(self):
        """
        Test the `__setattr__` method when the `value` has a wrong shape.
        """
        physical_property = PhysicalProperty(
            m_def=Section(
                iri='http://fairmat-nfdi.eu/taxonomy/PhysicalProperty', rank=[]
            )
        )
        physical_property.source = 'simulation'
        physical_property.variables = []
        # `physical_property.value` must have shape=[]
        value = np.ones((3, 3))
        wrong_shape = list(value.shape)
        with pytest.raises(ValueError) as exc_info:
            physical_property.value = value
        assert (
            str(exc_info.value)
            == f'The shape of the stored `value` {wrong_shape} does not match the full shape {physical_property.full_shape} extracted from the variables `n_grid_points` and the `shape` defined in `PhysicalProperty`.'
        )

    def test_setattr_none(self):
        """
        Test the `__setattr__` method when setting the `value` to `None`.
        """
        physical_property = PhysicalProperty(
            m_def=Section(
                iri='http://fairmat-nfdi.eu/taxonomy/PhysicalProperty', rank=[]
            )
        )
        physical_property.source = 'simulation'
        physical_property.variables = []
        with pytest.raises(ValueError) as exc_info:
            physical_property.value = None
        assert (
            str(exc_info.value)
            == f'The value of the physical property {physical_property.name} is None. Please provide a finite valid value.'
        )

    def test_is_derived(self):
        """
        Test the `normalize` and `_is_derived` methods.
        """
        section = SectionWithProperties()
        # Testing a directly parsed physical property
        not_derived_property = PhysicalProperty(
            m_def=Section(
                iri='http://fairmat-nfdi.eu/taxonomy/PhysicalProperty', rank=[]
            )
        )
        not_derived_property.source = 'simulation'
        section.physical_property.append(not_derived_property)
        assert not_derived_property._is_derived() is False
        not_derived_property.normalize(EntryArchive(), logger)
        assert not_derived_property.is_derived is False
        # Testing a derived physical property
        derived_property = PhysicalProperty(
            m_def=Section(
                iri='http://fairmat-nfdi.eu/taxonomy/PhysicalProperty', rank=[]
            )
        )
        section.physical_property.append(derived_property)
        derived_property.source = 'analysis'
        derived_property.physical_property_ref = not_derived_property
        assert derived_property._is_derived() is True
        derived_property.normalize(EntryArchive(), logger)
        assert derived_property.is_derived is True
