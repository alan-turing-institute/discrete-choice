"""
pylogit interface
"""

from . import Interface
from .. import MultinomialLogit
from collections import OrderedDict
import numpy as np
import pandas as pd
import pylogit as pl

# Column label for long format data indicating whether the row corresponds to
# the choice taken by the individual
_CHOICE_COL = 'choice_bool'
# Column label giving the unique number of the observation in the long format
_OBSERVATION_COL = 'observation_id'
# Column label giving the corresponding choice (encoded as a number) for each
# row in the long format
_CHOICE_ID_COL = 'choice_id'


class PylogitInterface(Interface):
    _valid_models = [MultinomialLogit]

    def __init__(self, model):
        super().__init__(model)

        if not isinstance(model.data, pd.DataFrame):
            raise NoDataLoaded

        # Create mapping from choice strings to integers begining from 1
        number_of_choices = model.number_of_choices()
        self.choice_encoding = dict(
            zip(model.choices, np.arange(number_of_choices, dtype=int)+1))

        self._convert_to_long_format()
        self._create_specification_and_names()
        self._create_model()

    def _encode_choices_as_integers(self):
        """
        Convert choice labels from strings to integers as pylogit expects
        """
        model = self.model
        choice_encoding = self.choice_encoding

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
        model.data[_CHOICE_COL] = model.data[model.choice_column].apply(
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
        self.model.data[_OBSERVATION_COL] = np.arange(model.data.shape[0],
                                                      dtype=int)+1

        # Use pylogit routine to convert to long format
        self.long_data = pl.convert_wide_to_long(
            wide_data=model.data,
            ind_vars=model.choice_independent_variables,
            alt_specific_vars=alt_specific_vars,
            availability_vars=availability_vars,
            obs_id_col=_OBSERVATION_COL,
            choice_col=_CHOICE_COL,
            new_alt_id_name=_CHOICE_ID_COL
            )

        # Remove choice bool and observation_id column from wide data
        model.data.drop(columns=[_CHOICE_COL, _OBSERVATION_COL],
                        inplace=True)

    def _create_specification_and_names(self):
        """
        Create the pylogit specification and paramter names dictionaries.
        """
        specification = OrderedDict()
        names = OrderedDict()
        pass

        model = self.model
        choice_encoding = self.choice_encoding

        # Intercepts
        specification['intercept'] = [choice_encoding[choice]
                                      for choice in model.intercepts.keys()]
        names['intercept'] = [intercept
                              for intercept in model.intercepts.values()]

        # Variables
        for variable in model.all_variables():
            # Identify which choice utilities contain variable
            relevant_choices = [
                choice for choice in model.choices
                if variable in model.specification[choice].all_variables]

            # Determine the corresponding parameters
            parameters = [model.specification[choice].term_dict[variable]
                          for choice in relevant_choices]

            # Group choices into parameter sets using choice encoding
            parameter_set = set(parameters)
            parameter_set = {parameter: [] for parameter in parameter_set}
            for choice, parameter in zip(relevant_choices, parameters):
                parameter_set[parameter].append(choice_encoding[choice])

            # Unpack choice lists of length 1
            for parameter, choices in parameter_set.items():
                if len(choices) == 1:
                    parameter_set[parameter] = choices[0]

            # Create specification dictionary entry
            specification[variable] = list(parameter_set.values())
            # Create names dictionary entry
            names[variable] = list(parameter_set.keys())

        self.specification = specification
        self.names = names

    def _create_model(self):
        """
        Create the pylogit model class
        """
        self.pylogit_model = pl.create_choice_model(
                data=self.long_data,
                alt_id_col=_CHOICE_ID_COL,
                obs_id_col=_OBSERVATION_COL,
                choice_col=_CHOICE_COL,
                specification=self.specification,
                names=self.names,
                model_type='MNL')

    def estimate(self, method='BFGS'):
        """
        Estimate the parameters of the choice model.
        """
        initial_parameters = np.zeros(self.model.number_of_parameters())
        self.pylogit_model.fit_mle(
            init_vals=initial_parameters,
            method=method)


class NoDataLoaded(Exception):
    """
    Exception for when it is attempted to create a pylogit interface from a
    model with no data.
    """
    def __init__(self):
        super().__init__('The model must be loaded with data before creating a'
                         ' pylogit interface')
