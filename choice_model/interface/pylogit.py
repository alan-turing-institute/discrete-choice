"""
pylogit interface
"""

from .interface import Interface, requires_estimation
from .. import MultinomialLogit
from collections import OrderedDict
from contextlib import redirect_stdout
from io import StringIO
import numpy as np
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
    """
    Pylogit interface class

    Args:
        model (ChoiceModel): The choice model to create an interface for.
    """
    _valid_models = [MultinomialLogit]
    name = 'pylogit'

    def __init__(self, model, **kwargs):
        super().__init__(model)

        # Create mapping from choice strings to integers begining from 1
        number_of_alternatives = model.number_of_alternatives()
        self.choice_encoding = dict(
            zip(model.alternatives,
                np.arange(number_of_alternatives, dtype=int)+1)
            )

        self._convert_to_long_format()
        self._create_specification_and_names()
        self._create_model()

    def _encode_alternatives_as_integers(self):
        """
        Convert choice labels from strings to integers as pylogit expects
        """
        model = self.model
        choice_encoding = self.choice_encoding

        # Create alternative specific variables dictionary using the integer
        # encoding
        alt_specific_vars = {}
        for variable in model.alternative_dependent_variables.keys():
            temp = model.alternative_dependent_variables[variable]
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
         availability_vars) = self._encode_alternatives_as_integers()
        # Create observation number column as a range of integers from 1
        model.data[_OBSERVATION_COL] = np.arange(model.data.shape[0],
                                                 dtype=int)+1

        # Use pylogit routine to convert to long format
        self.long_data = pl.convert_wide_to_long(
            wide_data=model.data,
            ind_vars=model.alternative_independent_variables,
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
            relevant_alternatives = [
                choice for choice in model.alternatives
                if variable in model.specification[choice].all_variables]

            # Determine the corresponding parameters
            parameters = [model.specification[choice].term_dict[variable]
                          for choice in relevant_alternatives]

            # Group alternatives into parameter sets using choice encoding
            parameter_set = set(parameters)
            parameter_set = {parameter: [] for parameter in parameter_set}
            for choice, parameter in zip(relevant_alternatives, parameters):
                parameter_set[parameter].append(choice_encoding[choice])

            # Unpack choice lists of length 1
            for parameter, alternatives in parameter_set.items():
                if len(alternatives) == 1:
                    parameter_set[parameter] = alternatives[0]

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
        Estimate the parameters of the choice model using pylogit.
        """
        initial_parameters = np.zeros(self.model.number_of_parameters())

        # Capture stdout as this contains the estimation time
        stdout = StringIO()
        with redirect_stdout(stdout):
            # Call the pylogit estimation routine
            self.pylogit_model.fit_mle(
                init_vals=initial_parameters,
                method=method)

        # Get estimation time from stdout
        self._estimation_time = float(
            stdout.getvalue().splitlines()[2].split()[-2])

        # Set estimated flag
        self._estimated = True

    @requires_estimation
    def display_results(self):
        self.pylogit_model.print_summaries()

    @requires_estimation
    def null_log_likelihood(self):
        return self.pylogit_model.null_log_likelihood

    @requires_estimation
    def final_log_likelihood(self):
        return self.pylogit_model.log_likelihood

    @requires_estimation
    def parameters(self):
        return dict(self.pylogit_model.params)

    @requires_estimation
    def standard_errors(self):
        return dict(self.pylogit_model.standard_errors)

    @requires_estimation
    def t_values(self):
        return dict(self.pylogit_model.tvalues)

    @requires_estimation
    def estimation_time(self):
        return self._estimation_time
