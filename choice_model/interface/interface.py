"""
Base interface class definition
"""

from .. import ChoiceModel
import pandas as pd


class Interface(object):
    _valid_models = [ChoiceModel]

    def __init__(self, model):
        self._ensure_valid_model(model)
        self.model = model

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
        Print the results of estimation
        """
        raise NotImplementedError(
            'display_results has not been implemented in this class')


class NoDataLoaded(Exception):
    """
    Exception for when it is attempted to create a pylogit interface from a
    model with no data.
    """
    def __init__(self):
        super().__init__('The model must be loaded with data before creating a'
                         ' pylogit interface')
