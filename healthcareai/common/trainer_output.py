"""
outputs.py is a decorator function that prints to stdout.

This eliminates lots of boilerplate code in superviseModelTrainer.
"""

import inspect
from functools import partial, wraps


def trainer_output(func=None, *, debug=False):
    """
    Trainer output decorator for functions that train models.

    This is a decorator that can be applied to any function, and it will print
    helpful information to the console such as the model type, and training
    results.

    Args:
        func (function): Function to be applied with decorator.
        debug (bool): Debug option true or false.
        * (params): trainer_output arguments.

    Returns:
        trained_model|function: returns trained_model when called without a
        function, or returns a callable when supplied with arguments.
    """

    if func is None:
        # This func is only None when extra arguments are supplied, return a
        # callable instead, which will get run and goes to the def wrap. Handy
        # way of using decorators with extra arguments.
        return partial(trainer_output, debug=debug)

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

        # If debug is true, output the function name, default, argument, and
        # returns.
        if debug:
            print("Function Name: {}, Function Defaults: {}, "
                  "Function Args: {} {}, Function Return: {}".format(
                func.__name__,
                inspect.signature(func),
                args,
                kwargs,
                trained_model))

        return trained_model

    return wrap
