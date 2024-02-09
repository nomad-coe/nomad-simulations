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

from nomad.datamodel.data import ArchiveSection


def get_sub_section_from_section_parent(
    section: ArchiveSection, sub_section: str, logger
) -> ArchiveSection:
    """
    Gets the sub_section of a section by performing a seesaw move, going to the parent of
    section, and then going down to sub_section. Example, if section is `Symmetry`, and we
    want to resolve `AtomicCell` (sub_section), this methods goes up to `ModelSystem` from `Symmetry`,
    and then goes down to `AtomicCell`.

    If the sub_section is a list, it returns the first element of the list. If the sub_section is
    a single section, it returns the section.

    Args:
        section (ArchiveSection): The section to check for its parent and retrieve the sub_section.
        sub_section (str): The name of the sub_section to retrieve from the parent.

    Returns:
        sub_section_sec (ArchiveSection): The sub_section to be returned.
    """
    if section.m_parent is None:
        logger.error("Could not find the parent of the section.")
        return
    if not section.m_parent.m_xpath(sub_section):
        logger.error("Could not find the section.m_parent.sub_section.")
        return
    sub_section_sec = getattr(section.m_parent, sub_section)
    if isinstance(sub_section_sec, list):
        if len(sub_section_sec) == 0:
            logger.error("The sub_section is empty.")
            return
        return sub_section_sec[0]
    elif isinstance(sub_section_sec, ArchiveSection):
        return sub_section_sec
    return
