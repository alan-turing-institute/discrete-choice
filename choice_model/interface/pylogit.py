"""
pylogit interface
"""

from . import Interface
from .. import MultinomialLogit
import numpy as np
import pandas as pd
import pylogit as pl


class PylogitInterface(Interface):
    _valid_models = [MultinomialLogit]

    def __init__(self, model):
        super().__init__(model)

        if not isinstance(model.data, pd.DataFrame):
            raise NoDataLoaded

        self._convert_to_long_format()

    def _encode_choices_as_integers(self):
        """
        Convert choice labels from strings to integers as pylogit expects
        """
        model = self.model
        number_of_choices = model.number_of_choices()
        # Create mapping from choice strings to integers begining from 0
        choice_encoding = dict(zip(model.choices, range(number_of_choices)))

        # Create alternative specific variables dictionary using the integer
        # encoding
        alt_specific_vars = {}
        for variable in model.choice_dependent_variables.keys():
            temp = model.choice_dependent_variables[variable]
            # Replace string choice key with integer key
            temp = {choice_encoding[key]: value for key, value in temp.items()}
            alt_specific_vars[variable] = temp

        # Create availaibility using the integer encoding
        availability_vars = {choice_encoding[key]: value
                             for key, value in model.availability.items()}

        # Create a dataframe column with the integer encoding. In the long
        # format this column will be a boolean for whether this choice was made
        # or not.
        model.data['choice_bool'] = model.data[model.choice_column].apply(
            lambda x: choice_encoding[x])

        return alt_specific_vars, availability_vars

    def _convert_to_long_format(self):
        """
        Convert data to the long format expected by pylogit
        """
        model = self.model
        (alt_specific_vars,
         availability_vars) = self._encode_choices_as_integers()
        # Create observation number column as a range of integers from 1
        self.model.data['observation_id'] = np.arange(model.data.shape[0],
                                                      dtype=int)+1

        # Use pylogit routine to convert to long format
        self.long_data = pl.convert_wide_to_long(
            wide_data=model.data,
            ind_vars=model.choice_independent_variables,
            alt_specific_vars=alt_specific_vars,
            availability_vars=availability_vars,
            obs_id_col='observation_id',
            choice_col='choice_bool',
            new_alt_id_name='choice_id'
            )

        # Remove choice bool and observation_id column from wide data
        model.data.drop(columns=['choice_bool', 'observation_id'],
                        inplace=True)


class NoDataLoaded(Exception):
    """
    Exception for when it is attempted to create a pylogit interface from a
    model with no data.
    """
    def __init__(self):
        super().__init__('The model must be loaded with data before creating a'
                         ' pylogit interface')
