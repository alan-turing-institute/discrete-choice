"""
Choice model definitions
"""

import pandas as pd
import yaml


class ChoiceModel(object):
    """
    Parent class for choice models
    """

    def __init__(self, title, choices, choice_column, availability, variables,
                 intercepts, parameters):

        self.title = title
        self.choices = choices
        self.choice_column = choice_column
        self.availability = availability
        self.variables = variables
        self.intercepts = intercepts
        self.parameters = parameters

        # Ensure all choices have an availability variable
        self._check_availability()

        # Ensure that there are enough intercepts (one fewer than the number of
        # choices)
        self._check_intercepts()

    @classmethod
    def from_yaml(cls, stream):
        """
        Read the model definition from a YAML file

        Args:
            stream (stream): Data stream of the model definition

        Returns:
            model (ChoiceModel): A choice model object corresponding to the
                definition in the stream
        """
        model_dict = yaml.load(stream)

        title = cls._copy_yaml_record('title', model_dict, stream)
        choices = cls._copy_yaml_record('choices', model_dict, stream)
        choice_column = cls._copy_yaml_record('choice_column', model_dict,
                                              stream)
        availability = cls._copy_yaml_record('availability', model_dict,
                                             stream)
        variables = cls._copy_yaml_record('variables', model_dict, stream)
        intercepts = cls._copy_yaml_record('intercepts', model_dict, stream)
        parameters = cls._copy_yaml_record('parameters', model_dict, stream)

        return cls(title, choices, choice_column, availability, variables,
                   intercepts, parameters)

    @staticmethod
    def _copy_yaml_record(key, yaml_dict, stream):
        if key in yaml_dict:
            return yaml_dict[key]
        else:
            raise MissingYamlKey(key, stream)

    def load_data(self, stream):
        """
        Load data into pandas dataframe.

        Args:
            stream (stream): csv data stream containing model data
        """
        self.data = pd.read_csv(stream)

        # Ensure that all required fields are defined in the dataframe
        self._check_fields(stream)

    def _check_fields(self, stream):
        """
        Ensures all required field are present in the pandas dataframe.
        """
        dataframe_columns = self.data.columns

        # Ensure choice column is present
        if self.choice_column not in dataframe_columns:
            raise MissingField(self.choice_column, stream)

        # Ensure all availability variables are present
        for availability in self.availability.values():
            if availability not in dataframe_columns:
                raise MissingField(availability, stream)

        # Ensure all variables are present
        for variable in self.variables:
            if variable not in dataframe_columns:
                raise MissingField(variable, stream)

    def _check_availability(self):
        for choice in self.choices:
            if choice not in self.availability:
                raise UndefinedAvailability(choice)

    def _check_intercepts(self):
        n_intercepts = len(self.intercepts)
        n_required = len(self.choices) - 1
        if n_intercepts != n_required:
            raise IncorrectNumberOfIntercepts(n_intercepts, n_required)


class MultinomialLogit(ChoiceModel):
    """
    Multinomial logit choice model class
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def from_yaml(self, stream):
        super().__init__()


class MissingYamlKey(Exception):
    """
    Exception for missing, required YAML keys.
    """
    def __init__(self, key, stream):
        super().__init__(
            'Required key "{}" missing in file "{}"'.format(key,
                                                            stream.name)
            )


class UndefinedAvailability(Exception):
    """
    Exception for an undefined availability variable.
    """
    def __init__(self, choice):
        super().__init__(
            'Availability variable for choice "{}" not defined'.format(choice)
            )


class MissingField(Exception):
    """
    Exception for missing field in the data file
    """
    def __init__(self, field, stream):
        super().__init__(
            'Field "{}" not present in data file "{}"'.format(
                field,
                stream.name)
            )


class IncorrectNumberOfIntercepts(Exception):
    """
    Exception for when the number of declared intercepts is incompatible
    """
    def __init__(self, number, required_number):
        super().__init__(
            'Number of intercepts defined ({}) != number required ({})'.format(
                number, required_number)
            )
