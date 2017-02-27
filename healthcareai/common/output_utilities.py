import json
import pickle

def save_output_to_json(filename,output):
    with open(filename, 'w') as open_file:
        json.dump(output, open_file, indent=4, sort_keys=True)

def save_best_estimator_to_pickle(filename,best_estimator):
    with open(filename + '.pkl', 'wb') as open_file:
        pickle.dump(best_estimator, open_file)

