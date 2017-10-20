"""
outputs.py is a decorator function that prints to stdout.

This eliminates lots of boilerplate code in superviseModelTrainer.
"""

import inspect
from functools import partial, wraps


def trainer_output(func):
    """
    Trainer output decorator for functions that train models.

    This is a decorator that can be applied to any function, and it will print
    helpful information to the console such as the model type, and training
    results.

    Args:
        func (function): Function to be applied with decorator.

    Returns:
        trained_model: returns trained_model
    """

    # Wrap around our function so that if debug is true, we can print out
    # inputs and outputs. The @wraps decorator copies the parent function's
    # attributes, such as __name__, and input parameters.
    @wraps(func)
    def wrap(self, *args, **kwargs):
        # Since we have decorated the function and self at runtime, we can get
        # the name of the model, and construct the name out of the function
        # name. Then use self's model type to output the model type (regression
        # or classification)

        algorithm_name = " ".join(func.__name__.split("_")).title()
        print("Training: {} , Type: {}".format(
            algorithm_name,
            self._advanced_trainer.model_type))
        trained_model = func(self, *args, **kwargs)
        trained_model.print_training_results()
        return trained_model

    return wrap
