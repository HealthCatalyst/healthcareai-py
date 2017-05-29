import json
import pickle

from healthcareai.common.healthcareai_error import HealthcareAIError


def save_dict_object_to_json(dictionary, filename):
    """
    Save a dictionary object as json

    Args:
        dictionary (dict): the dictionary to save as a JSON file 
        filename (str): file name to save the JSON to 
    """
    _validate_filename_is_string(filename)

    with open(filename, 'w') as open_file:
        json.dump(dictionary, open_file, indent=4, sort_keys=True)


def save_object_as_pickle(object_to_pickle, filename):
    """
    Saves a python object of any type to a pickle file with the given filename
    
    Args:
        object_to_pickle (object): the object to save to disk
        filename (str): file name to save the object to
        
    """
    _validate_filename_is_string(filename)

    with open(filename, 'wb') as open_file:
        pickle.dump(object_to_pickle, open_file)


def load_pickle_file(filename):
    """
    Loads a python object of any type from a pickle file with the given filename

    Args:
        filename (str): File name to load 

    Returns:
        (object): A python object
    """
    _validate_filename_is_string(filename)

    try:
        with open(filename, 'rb') as open_file:
            return pickle.load(open_file)
    except FileNotFoundError as e:
        raise HealthcareAIError(
            'No file named \'{}\' was found. Please verify the file you intend to load'.format(filename))


def load_saved_model(filename, debug=True):
    """
    Convenience method for a simple API without users needing to know what pickling is. Also prints model metadata
    
    Args:
        filename (str): name of saved file to laod 
        debug (bool): Print debug output to console by default

    Returns:
        (TrainedSupervisedModel): a saved model
    """
    _validate_filename_is_string(filename)

    trained_model = load_pickle_file(filename)

    if debug:
        print('Trained model loaded from file: {}\n    Type: {}'.format(filename, type(trained_model)))
        if hasattr(trained_model, 'model'):
            print('    Model type: {}'.format(type(trained_model.model)))

    return trained_model


def _validate_filename_is_string(filename):
    """ Validates the a parameter is a string and returns a helpful error message if it is not. """
    if not isinstance(filename, str):
        raise HealthcareAIError('Filename must be a string. You passed in a {}'.format(filename))


if __name__ == "__main__":
    pass
