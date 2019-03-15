"""
Base interface class definition
"""

from .. import ChoiceModel
from functools import wraps
import pandas as pd


class Interface(object):
    _valid_models = [ChoiceModel]

    def __init__(self, model):
        self._ensure_valid_model(model)
        self.model = model
        self._estimated = False

        if not isinstance(model.data, pd.DataFrame):
            raise NoDataLoaded

    @classmethod
    def _ensure_valid_model(cls, model):
        if type(model) not in cls._valid_models:
            raise TypeError(
                'Argument "model" for cls.__name__ must be one of {}'.format(
                    [model_class.__name__ for model_class in cls._valid_models]
                    )
                )

    def estimate(self):
        """
        Estimate the parameters of the choice model.
        """
        raise NotImplementedError(
            'estimate has not been implemented in this class')

    def display_results(self):
        """
        Print the results of estimation.
        """
        raise NotImplementedError(
            'display_results has not been implemented in this class')

    def null_log_likelihood(self):
        """
        Determine the null log likelihood of the model.

        Returns:
            (float): The null log likelihood (i.e. when all parameters are
                zero).
        """
        raise NotImplementedError(
            'null_log_likelihood has not been implemented in this class')

    def final_log_likelihood(self):
        """
        Determine the optimised log likelihood of the model.

        Returns:
            (float): The log likelihood with optimised parameters.
        """
        raise NotImplementedError(
            'final_log_likelihood has not been implemented in this class')

    def parameters(self):
        """
        Determine the optimised parameters of the model.

        Returns:
            (dict): A dictionary of optimised parameters. The keys are the
                parameter names defined in the model, the values are the
                optimised parameters.
        """
        raise NotImplementedError(
            'parameters has not been implemented in this class')

    def standard_errors(self):
        """
        Determine the standard errors of the optimised parameters.

        Returns:
            (dict): A dictionary of standard errors. The keys are the parameter
                names defined in the model, the values are the optimised
                parameters.
        """
        raise NotImplementedError(
            'standard_errors has not been implemented in this class')

    def t_values(self):
        """
        Determine the t values of the optimised parameters.

        Returns:
            (dict): A dictionary of t values. The keys are the parameter
                names defined in the model, the values are the optimised
                parameters.
        """
        raise NotImplementedError(
            't_values has not been implemented in this class')

    def estimation_time(self):
        """
        Report the estimation time for the interface in seconds

        Returns:
            (float): The estimation time in seconds.
        """
        raise NotImplementedError(
            'estimation_time has not been implemented in this class')


class NoDataLoaded(Exception):
    """
    Exception for when it is attempted to create a pylogit interface from a
    model with no data.
    """
    def __init__(self):
        super().__init__('The model must be loaded with data before creating a'
                         ' pylogit interface')


def requires_estimation(method):
    """
    Decorator to assert that model paramters must have been estimated before
    calling a method.
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if self._estimated:
            return method(self, *args, **kwargs)
        else:
            raise NotEstimated
    return wrapper


class NotEstimated(Exception):
    """
    Exception raised when a method requires estimation to have been conducted
    but it has not yet.
    """
    def __init__(self):
        super().__init__(
            'The model parameters must have been estimated to call this method'
            )
