"""
Choice model definitions
"""

import pandas as pd
import yaml


class ChoiceModel(object):
    """
    Parent class for choice models
    """

    def __init__(self, title, data_file, choices, choice_column,
                 availability, variables, intercepts, parameters):

        self.title = title
        self.data_file = data_file
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

        # Load data
        self.load_data()

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
        data_file = cls._copy_yaml_record('data', model_dict, stream)
        choices = cls._copy_yaml_record('choices', model_dict, stream)
        choice_column = cls._copy_yaml_record('choice_column', model_dict,
                                              stream)
        availability = cls._copy_yaml_record('availability', model_dict,
                                             stream)
        variables = cls._copy_yaml_record('variables', model_dict, stream)
        intercepts = cls._copy_yaml_record('intercepts', model_dict, stream)
        parameters = cls._copy_yaml_record('parameters', model_dict, stream)

        return cls(title, data_file, choices, choice_column,
                   availability, variables, intercepts, parameters)

    @staticmethod
    def _copy_yaml_record(key, yaml_dict, stream):
        if key in yaml_dict:
            return yaml_dict[key]
        else:
            raise MissingYamlKey(key, stream)

    def load_data(self):
        """
        Load data into pandas dataframe.
        """
        with open(self.data_file, 'r') as data_file:
            self.data = pd.read_csv(data_file)

        # Ensure that all required fields are defined in the dataframe
        self._check_fields()

    def _check_fields(self):
        """
        Ensures all required field are present in the pandas dataframe.
        """
        dataframe_columns = self.data.columns

        # Ensure choice column is present
        if self.choice_column not in dataframe_columns:
            raise MissingField(self.choice_column, self.data_file)

        # Ensure all availability variables are present
        for availability in self.availability.values():
            if availability not in dataframe_columns:
                raise MissingField(availability, self.data_file)

        # Ensure all variables are present
        for variable in self.variables:
            if variable not in dataframe_columns:
                raise MissingField(variable, self.data_file)

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
                                                            stream.file)
            )


class UndefinedAvailability(Exception):
    """
    Exception for an undefined availability variable.
    """
    def __init__(self, choice, stream):
        super().__init__(
            'Availability variable for choice "{}" not defined'.format(
                choice, stream.file)
            )


class MissingField(Exception):
    """
    Exception for missing field in the data file
    """
    def __init__(self, field, data_file):
        super().__init__(
            'Field "{}" not present in data file "{}"'.format(
                field,
                data_file)
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
