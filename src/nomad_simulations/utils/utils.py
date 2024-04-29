#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD.
# See https://nomad-lab.eu for further info.
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

from math import factorial
from typing import Optional
from structlog.stdlib import BoundLogger

from nomad.datamodel.data import ArchiveSection


def get_sibling_section(
    section: ArchiveSection,
    sibling_section_name: str,
    index_sibling: int = 0,
    logger: BoundLogger = None,
) -> Optional[ArchiveSection]:
    """
    Gets the sibling section of a section by performing a seesaw move by going to the parent
    of the section and then going down to the sibling section. This is used, e.g., to get
    the `AtomicCell` section from the `Symmetry` section and by passing through the `ModelSystem`.

    Example of the sections structure:

        parent_section
          |__ section
          |__ sibling_section


    If the sibling_section is a list, it returns the element `index_sibling` of that list. If
    the sibling_section is a single section, it returns the sibling_section itself.

    Args:
        section (ArchiveSection): The section to check for its parent and retrieve the sibling_section.
        sibling_section (str): The name of the sibling_section to retrieve from the parent.
        index_sibling (int): The index of the sibling_section to retrieve if it is a list.
        logger (BoundLogger): The logger to log messages.

    Returns:
        sibling_section (ArchiveSection): The sibling_section to be returned.
    """
    if not sibling_section_name:
        logger.warning('The sibling_section_name is empty.')
        return None
    sibling_section = section.m_xpath(f'm_parent.{sibling_section_name}', dict=False)
    # If the sibling_section is a list, return the element `index_sibling` of that list
    if isinstance(sibling_section, list):
        if index_sibling >= len(sibling_section):
            logger.warning('The index of the sibling_section is out of range.')
            return None
        return sibling_section[index_sibling]
    return sibling_section


# ? Check if this utils deserves its own file after extending it
class RussellSaundersState:
    @classmethod
    def generate_Js(cls, J1: float, J2: float, rising=True):
        J_min, J_max = sorted([abs(J1), abs(J2)])
        generator = range(
            int(J_max - J_min) + 1
        )  # works for both for fermions and bosons
        if rising:
            for jj in generator:
                yield J_min + jj
        else:
            for jj in generator:
                yield J_max - jj

    @classmethod
    def generate_MJs(cls, J, rising=True):
        generator = range(int(2 * J + 1))
        if rising:
            for m in generator:
                yield -J + m
        else:
            for m in generator:
                yield J - m

    def __init__(self, *args, **kwargs):
        self.J = kwargs.get('J')
        if self.J is None:
            raise TypeError
        self.occupation = kwargs.get('occ')
        if self.occupation is None:
            raise TypeError

    @property
    def multiplicity(self):
        return 2 * self.J + 1

    @property
    def degeneracy(self):
        return factorial(self.multiplicity) / (
            factorial(self.multiplicity - self.occupation) * factorial(self.occupation)
        )


def is_not_representative(model_system, logger: BoundLogger = None):
    """
    Checks if the given `ModelSystem` is not representative and logs a warning.

    Args:
        model_system (ModelSystem): The `ModelSystem` to check.
        logger (BoundLogger): The logger to log messages.

    Returns:
        (bool): True if the `ModelSystem` is not representative, False otherwise.
    """
    if not model_system.is_representative:
        logger.warning('The `ModelSystem` was not found to be representative.')
        return True
    return False


def check_archive(archive: ArchiveSection, logger: BoundLogger = None):
    """
    Checks if the given `Archive` is empty and logs a warning.

    Args:
        archive (ArchiveSection): The `ArchiveSection` to check.
        logger (BoundLogger): The logger to log messages.

    Returns:
        (bool): True if the archive is empty, False otherwise.
    """
    if not archive:
        logger.warning('The `archive` is empty.')
        return False
    return True
