"""
Validators.

This module contains simple functions that can be placed as guards on any
methods you want to protect with nice HealthcareAIErrors.
"""
from pandas import DataFrame

from healthcareai.common.healthcareai_error import HealthcareAIError


def validate_dataframe_input_for_transformer(possible_dataframe):
    """Raise an error if input is not a pandas dataframe."""
    _validate_dataframe_input(possible_dataframe, 'transformer')


def validate_dataframe_input_for_method(possible_dataframe):
    """Raise an error if input is not a pandas dataframe."""
    _validate_dataframe_input(possible_dataframe, 'method')


def _validate_dataframe_input(
        possible_dataframe,
        error_message_method_type='method'):
    """
    Raise an error if input is not a pandas dataframe.

    Args:
        possible_dataframe (object): incoming object to be validated
        error_message_method_type (str): short string that helps clarify error
    """
    if is_dataframe(possible_dataframe) is False:
        raise HealthcareAIError(
            'This {} requires a pandas dataframe and you passed in a {}'
            .format(
                error_message_method_type,
                type(possible_dataframe)))


def is_dataframe(possible_dataframe):
    """Return true if input is a pandas dataframe."""
    return issubclass(DataFrame, type(possible_dataframe))
