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

import pytest
import numpy as np

from nomad.datamodel import EntryArchive

from . import logger
from nomad_simulations.utils import (
    get_sibling_section,
    is_not_representative,
    check_archive,
)
from nomad_simulations.model_system import ModelSystem, AtomicCell, Symmetry


def test_get_sibling_section():
    """
    Test the `get_sibling_section` utility function.
    """
    parent_section = ModelSystem()
    section = AtomicCell(type='original')
    parent_section.atomic_cell.append(section)
    sibling_section = Symmetry()
    parent_section.symmetry.append(sibling_section)
    assert get_sibling_section(section, '', logger) is None
    assert get_sibling_section(section, 'symmetry', logger) == sibling_section
    assert get_sibling_section(sibling_section, 'atomic_cell', logger) == section
    assert get_sibling_section(section, 'symmetry', logger, index_sibling=2) is None
    section2 = AtomicCell(type='primitive')
    parent_section.atomic_cell.append(section2)
    assert (
        get_sibling_section(sibling_section, 'atomic_cell', logger, index_sibling=0)
        == section
    )
    assert (
        get_sibling_section(sibling_section, 'atomic_cell', logger, index_sibling=1)
        == section2
    )


def test_is_not_representative():
    """
    Test the `is_not_representative` utility function.
    """
    assert is_not_representative(None, logger) is None
    assert is_not_representative(ModelSystem(), logger)
    assert not is_not_representative(ModelSystem(is_representative=True), logger)


def test_check_archive():
    """
    Test the `check_archive` utility function.
    """
    assert not check_archive(None, logger)
    assert check_archive(EntryArchive(), logger)
