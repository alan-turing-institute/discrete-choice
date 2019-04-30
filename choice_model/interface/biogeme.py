"""
Biogeme interface class
"""

from .interface import Interface, requires_estimation # noqa
from .. import MultinomialLogit
import biogeme.biogeme as bio # noqa
from biogeme.expressions import Beta, Variable, bioLogLogit
import biogeme.database as biodb
import numpy as np
import os
import sys

_CHOICE_COL = 'choice_int'


class BiogemeInterface(Interface):
    """
    Biogeme interface class.

    Args:
        model (ChoiceModel): The choice model to create a Biogeme interface
            for.
    """
    _valid_models = [MultinomialLogit]
    name = 'Biogeme'

    def __init__(self, model):
        super().__init__(model)

        # Create mapping from choice strings to integers begining from 1
        number_of_alternatives = model.number_of_alternatives()
        self.choice_encoding = dict(
            zip(model.alternatives,
                np.arange(number_of_alternatives, dtype=int)+1)
            )

        self.data = model.data.copy(deep=True)
        # Add choice column using integer encoding
        self.data[_CHOICE_COL] = self.data[model.choice_column].apply(
            lambda x: self.choice_encoding[x]
            )
        # Drop original choice column
        self.data.drop(columns=[model.choice_column], inplace=True)

        # Create biogeme database. This command creates a file in the current
        # working directory called 'headers.py'(!?) that contains some required
        # functions.
        database = biodb.Database("data", self.data)

        # Import the headers file.
        # with cwd_path():
        #     from headers import Beta

        # Define parameters
        # The 'Beta' function defines information about a parameter. Its
        # arguments are,
        # 1. A string giving the name of the parameter
        # 2. The default value (here always 0)
        # 3. The lower bound (here always None)
        # 4. The upper bound (here always None)
        # 5. A flag, 0 if the variable is to be estimated, 1 if not.
        parameters = {}
        for parameter in model.parameters:
            parameters[parameter] = Beta(parameter, 0, None, None, 0)
        # Add intercepts
        for intercept in model.intercepts.values():
            parameters[intercept] = Beta(intercept, 0, None, None, 0)

        # Define variables
        variables = {}
        for variable in model.all_variable_fields():
            variables[variable] = Variable(variable)

        # Add choice column
        variables[_CHOICE_COL] = Variable(_CHOICE_COL)

        # Define availabilities
        availabilities = {}
        for alternative, availability in model.availability.items():
            availabilities[self.choice_encoding[alternative]] = Variable(
                availability)

        # Define utility functions using the integer encoding
        v = {}
        # Loop over utility objects for each alternatve
        for alternative, specification in model.specification.items():
            alt_id = self.choice_encoding[alternative]
            # Add products of variables and parameters to utility
            variable, parameter = specification.terms[0]
            if variable in model.alternative_dependent_variables:
                variable = model.alternative_dependent_variables[variable][
                    alternative]
            v[alt_id] = (parameters[parameter] * variables[variable])
            for variable, parameter in specification.terms[1:]:
                if variable in model.alternative_dependent_variables:
                    variable = model.alternative_dependent_variables[variable][
                        alternative]
                v[alt_id] += (parameters[parameter] * variables[variable])

            # Add intercept to utility (if there is one)
            if specification.intercept is not None:
                v[alt_id] += parameters[specification.intercept]

        # Create biogeme model object
        problem = bioLogLogit(v, availabilities, variables[_CHOICE_COL])
        self.biogeme_model = bio.BIOGEME(database, problem)
        self.biogeme_model.modelName = model.title

    def estimate(self):
        """
        Estimate the parameters of the choice model using Biogeme
        """

        # Call biogeme estimation routine and store results
        self.results = self.biogeme_model.estimate()

        # Set estimated flag
        self._estimated = True

    @requires_estimation
    def display_results(self):
        print(self.results.getEstimatedParameters())

    @requires_estimation
    def null_log_likelihood(self):
        return self.results.getGeneralStatistics()['Init log likelihood'][0]

    @requires_estimation
    def final_log_likelihood(self):
        return self.results.getGeneralStatistics()['Final log likelihood'][0]

    @requires_estimation
    def parameters(self):
        return self.results.getEstimatedParameters()['Value'].as_dict()

    @requires_estimation
    def standard_errors(self):
        return self.results.getEstimatedParameters()['Std err'].as_dict()

    @requires_estimation
    def t_values(self):
        return self.results.getEstimatedParameters()['t-test'].as_dict()

    @requires_estimation
    def estimation_time(self):
        delta = self.results.getGeneralStatistics()['Optimization time'][0]
        return delta.total_seconds()


class cwd_path(object):
    def __init__(self):
        self.path = os.getcwd()

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        sys.path.remove(self.path)
