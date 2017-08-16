import sys

from healthcareai.common.healthcareai_error import HealthcareAIError


def validate_pyodbc_is_loaded():
    """ Simple check that alerts user if they are do not have pyodbc installed, which is not a requirement. """
    if 'pyodbc' not in sys.modules:
        raise HealthcareAIError('Using this function requires installation of pyodbc.')


def validate_sqlite3_is_loaded():
    """ Simple check that alerts user if they are do not have sqlite installed, which is not a requirement. """
    if 'sqlite3' not in sys.modules:
        raise HealthcareAIError('Using this function requires installation of sqlite3.')
