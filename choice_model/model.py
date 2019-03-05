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
        for choice in choices:
            if choice not in self.availability:
                raise UndefinedAvailability(choice)

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

        if 'title' in model_dict:
            title = model_dict['title']
        else:
            raise MissingYamlKey('title', stream)

        if 'data' in model_dict:
            data_file = model_dict['data']
        else:
            raise MissingYamlKey('data', stream)

        if 'choices' in model_dict:
            choices = model_dict['choices']
        else:
            raise MissingYamlKey('choices', stream)

        if 'choice_column' in model_dict:
            choice_column = model_dict['choice_column']
        else:
            raise MissingYamlKey('choice_column', stream)

        if 'availability' in model_dict:
            availability = model_dict['availability']
        else:
            raise MissingYamlKey('availability', stream)

        if 'variables' in model_dict:
            variables = model_dict['variables']
        else:
            raise MissingYamlKey('variables', stream)

        if 'intercepts' in model_dict:
            intercepts = model_dict['intercepts']
        else:
            raise MissingYamlKey('intercepts', stream)

        if 'parameters' in model_dict:
            parameters = model_dict['parameters']
        else:
            raise MissingYamlKey('parameters', stream)

        return cls(title, data_file, choices, choice_column,
                   availability, variables, intercepts, parameters)

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
