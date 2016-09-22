from os import path


def fixture(file):
    """ Returns the absolute path for a fixture that contains sample data.

    Parameters
    ----------
    file (str) : file name of the fixture.

    Returns
    -------
    String representing the absolute path of the fixture file.

    Examples
    --------
    >>> fixture('SampleData.csv')
    """
    return path.join(path.abspath(path.dirname(__file__)),
                     '../hcpytools',
                     file)
