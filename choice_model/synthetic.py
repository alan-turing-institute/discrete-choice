"""
Routines for creating synthetic models and corresponding data
"""

from .model import MultinomialLogit
import numpy as np
import numpy.random as random
import pandas as pd


def synthetic_model(title, number_of_alternatives, number_of_variables):
    """
    Create a synthetic model. In the model produced, the utility function for
    each alternative is simply a linear combination of all variables, there are
    no choice independent variables and all alternatives are available in each
    record.

    Args:
        title (str): Title for the model object.
        number_of_alternatives (int): Number of alternatives to generate.
        number_of_variables (int): Number of variables (and also parameters) to
            generate.

    Returns:
        (MultinomialLogit): Multinomial logit choice model object.
    """
    # Define alternatives in the format alternative1, alternative2, etc.
    alternatives = ['alternative{}'.format(number)
                    for number in range(1, number_of_alternatives+1)]
    # Define availability columns in the format availability1, availability2,
    # etc.
    availability = {alternative: 'availability{}'.format(number+1)
                    for number, alternative in enumerate(alternatives)}
    # Define choice dependent variable columns in the format
    # alternative1_variable1, alternative1_variable2, etc.
    variables = {}
    for number in range(1, number_of_variables+1):
        variables['variable{}'.format(number)] = {
            alternative: '{}_variable{}'.format(alternative, number)
            for alternative in alternatives
            }
    # Define intercept names in the format c1, c1, etc.
    intercepts = {alternative: 'c{}'.format(number+1)
                  for number, alternative in enumerate(alternatives[:-1])}
    # Define parameters in the format parameter1, parameter2, etc.
    parameters = ['parameter{}'.format(number)
                  for number in range(1, number_of_variables+1)]

    # Create linear combination terms i.e. parameter1*variable1 +
    # parameter2*variable2 + ...
    all_variables = variables.keys()
    products = ['*'.join(pair) for pair in zip(parameters, all_variables)]
    linear_combination = ' + '.join(products)
    # Construct utility function strings. The last alternative does not have an
    # intercept
    specification = {}
    for alternative in alternatives[:-1]:
        specification[alternative] = (
            ' + '.join([intercepts[alternative], linear_combination])
            )
    specification[alternatives[-1]] = linear_combination

    model = MultinomialLogit(
        title=title,
        choices=alternatives,
        choice_column='choice',
        availability=availability,
        choice_independent_variables=[],
        choice_dependent_variables=variables,
        intercepts=intercepts,
        parameters=parameters,
        specification=specification
        )
    return model


def synthetic_data(model, number_of_records):
    """
    Generate synthetic data for a model.

    Args:
        model (ChoiceModel): The choice model object to create synthetic
            observatiosn for.
        number_of_records (int): The number of synthetic observations to
            create.

    Returns:
        (DataFrame): A pandas dataframe of synthetic data that can be
            loaded into model.
    """
    # Create dataframe with the necessary column labels
    data = pd.DataFrame(
        columns=(model.all_variable_fields() +
                 model.availability_fields() +
                 [model.choice_column])
        )

    # Populate the choice column with choices picked uniformly from the
    # models alternatives
    alternatives = model.choices
    data[model.choice_column] = random.choice(alternatives,
                                              size=number_of_records)

    # Set all availability columns to 1 (available)
    for column in model.availability_fields():
        data[column] = np.full(shape=number_of_records,
                               fill_value=1)

    # Fill all variable columns with uniform random numbers in the range
    # [0,1)
    for column in model.all_variable_fields():
        data[column] = random.random(size=number_of_records)

    return data
