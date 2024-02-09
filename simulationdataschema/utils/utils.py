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

from typing import Optional
from structlog.stdlib import BoundLogger

from nomad.datamodel.data import ArchiveSection


def get_sibling_section(
    section: ArchiveSection, sibling_section_name: str, logger: BoundLogger
) -> ArchiveSection:
    """
    Gets the sibling section of a section by performing a seesaw move by going to the parent
    of the section and then going down to the sibling section. This is used, e.g., to get
    the `AtomicCell` section from the `Symmetry` section and by passing through the `ModelSystem`.

    Example of the sections structure:

        parent_section
          |__ section
          |__ sibling_section


    If the sibling_section is a list, it returns the first element of the list. If the sibling_section is
    a single section, it returns the sibling_section itself.

    Args:
        section (ArchiveSection): The section to check for its parent and retrieve the sibling_section.
        sibling_section (str): The name of the sibling_section to retrieve from the parent.
        logger (BoundLogger): The logger to log messages.

    Returns:
        sibling_section (ArchiveSection): The sibling_section to be returned.
    """
    if section.m_parent is None:
        logger.warning("Could not find the parent of the section.")
        return
    if not section.m_parent.m_xpath(sibling_section_name):
        logger.warning("Could not find the section.m_parent.sub_section.")
        return
    sibling_section = getattr(section.m_parent, sibling_section_name)
    if isinstance(sibling_section, list):
        if len(sibling_section) == 0:
            logger.warning("The sub_section is empty.")
            return
        return sibling_section[0]  # ? extend for any section targeted as input
    elif isinstance(sibling_section, ArchiveSection):
        return sibling_section
    return
