import json
import pickle


def save_dict_object_to_json(filename, dictionary):
    """
    Save a dictionary object as json
    :param filename:
    :param dictionary:
    """
    with open(filename, 'w') as open_file:
        json.dump(dictionary, open_file, indent=4, sort_keys=True)


def save_object_as_pickle(filename, object_to_pickle):
    """
    Saves a python object of any type to a pickle file with the given filename
    :param filename:
    :param object_to_pickle:
    """
    with open(filename + '.pkl', 'wb') as open_file:
        pickle.dump(object_to_pickle, open_file)
