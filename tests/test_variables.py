import pytest
from nomad.datamodel import EntryArchive

from nomad_simulations.schema_packages.variables import Variables

from . import logger


class TestVariables:
    """
    Test the `Variables` class defined in `variables.py`.
    """

    @pytest.mark.parametrize(
        'n_points, points, result',
        [
            (3, [-1, 0, 1], 3),
            (5, [-1, 0, 1], 3),
            (None, [-1, 0, 1], 3),
            (4, None, 4),
            (4, [], 4),
        ],
    )
    def test_normalize(self, n_points: int, points: list, result: int):
        """
        Test the `normalize` and `get_n_points` methods.
        """
        variable = Variables(
            name='variable_1',
            n_points=n_points,
            points=points,
        )
        assert variable.get_n_points(logger) == result
        variable.normalize(EntryArchive(), logger)
        assert variable.n_points == result
