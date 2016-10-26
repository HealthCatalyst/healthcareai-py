from os import path


def fixture(file):
    """ Returns the absolute path for a fixtures that contains sample data.

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
