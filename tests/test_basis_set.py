from typing import Any, Optional

import pytest
from . import logger
from nomad.units import ureg
import numpy as np

from tests.conftest import refs_apw

from nomad_simulations.schema_packages.basis_set import (
    APWBaseOrbital,
    APWOrbital,
    APWLocalOrbital,
    APWLChannel,
    APWPlaneWaveBasisSet,
    BasisSet,
    BasisSetContainer,
    MuffinTinRegion,
    generate_apw,
)


def test_cutoff():
    """Test the quantitative results when computing certain plane-wave cutoffs."""
    p_unit = '1 / angstrom'
    ref_cutoff_radius = 1.823 * ureg(p_unit)
    pw = APWPlaneWaveBasisSet(cutoff_energy=500 * ureg('eV'))
    assert np.isclose(
        pw.cutoff_radius.to(p_unit).magnitude, ref_cutoff_radius.magnitude, atol=1e-3
    )  # reference computed by ChatGPT 4o

    pw.set_cutoff_fractional(1 / ref_cutoff_radius, logger)
    assert np.isclose(pw.cutoff_fractional, 1, rtol=1e-2)


def test_cutoff_failure():
    """Test modes where `cutoff_fractional` is not computed."""
    # missing cutoff_energy
    pw = APWPlaneWaveBasisSet()
    pw.set_cutoff_fractional(ureg.angstrom, logger)
    assert pw.cutoff_fractional is None

    # missing mt_radius
    pw = APWPlaneWaveBasisSet(cutoff_energy=500 * ureg('eV'))
    pw.set_cutoff_fractional(None, logger)
    assert pw.cutoff_fractional is None

    # cutoff_fractional already set
    pw = APWPlaneWaveBasisSet(cutoff_energy=500 * ureg('eV'), cutoff_fractional=1)
    pw.set_cutoff_fractional(ureg.angstrom, logger)
    assert pw.cutoff_fractional == 1


@pytest.mark.parametrize(
    'ref_index, species_def, cutoff',
    [
        (0, {}, None),
        (1, {}, 500.0),
        (2, {'H': {'r': 1, 'l_max': 2, 'orb_type': ['apw']}}, 500.0),
    ],
)
def test_full_apw(
    ref_index: int, species_def: dict[str, dict[str, Any]], cutoff: Optional[float]
):
    """Test the composite structure of APW basis sets."""
    assert (
        generate_apw(species_def, cutoff=cutoff).m_to_dict() == refs_apw[ref_index]
    )  # TODO: add normalization?
