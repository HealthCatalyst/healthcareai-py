"""outputs.py contains functions that prints to stdout."""

import inspect
from functools import partial, wraps


def trainer_output(func=None, *, debug=False):
    """Trainer output decorator for functions that train models.

    This is a decorator that can be applied to any functions, and it will output the model type, and training results.

    Args:
        func (function): Function to be applied with decorator.
        debug (bool): Debug option true or false.
        * (params): trainer_output arguments.

    Returns:
        trained_model, or function: returns trained_model when calls the function with no supplied function, and
            returns a callable when supplied with arguments.

    """

    # This func is only None when extra arguments are supplied, we will return a callable instead, which will get run
    # and goes to the def wrap. Handy way of using decorators with extra arguments.
    if func is None:
        return partial(trainer_output, debug=debug)

    # Wrap around our function so that if debug is true, so we can print out inputs and outputs. The @wraps decorator
    # helps copies the parent function's information, such as __name__, and input parameters.
    @wraps(func)
    def wrap(self, *args, **kwargs):
        # Since we have decorated function and self at runtime, we can get the name of the model, and construct the name
        # out of the function name. Then use self's model type to output the model type (regression or classification)
        print("Training:" + " ".join(func.__name__.split("_")).title() + ", Type:" + self._advanced_trainer.model_type)
        trained_model = func(self, *args, **kwargs)
        trained_model.print_training_results()

        # If debug is true, we can output the function name, default, argument, and returns.
        if debug:
            print("Function Name: {}, Function Defaults: {}, "
                  "Function Args: {} {}, Function Return: {}".format(func.__name__, inspect.signature(func),
                                                                     args, kwargs, trained_model))
        return trained_model

    return wrap
