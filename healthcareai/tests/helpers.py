"""Test helpers."""
from os import path


def fixture(file):
    # TODO deprecate after a better test for impact coding is devised.
    """Return the absolute path for a fixtures that contains sample data.

    Parameters
    ----------
    file (str) : file name of the fixtures.

    Returns
    -------
    String representing the absolute path of the fixtures file.

    Examples
    --------
    >>> fixtures('SampleData.csv')
    """
    return path.join(path.abspath(path.dirname(__file__)),
                     'fixtures',
                     file)


def assertBetween(self, minimum, maximum, value):
    """Fail if value is not between min and max (inclusive)."""
    self.assertGreaterEqual(value, minimum)
    self.assertLessEqual(value, maximum)
