"""
Choice model definitions.
"""

import pandas as pd
import yaml


class ChoiceModel(object):
    """
    Parent class for choice models.
    """

    def __init__(self, title, choices, choice_column, availability, variables,
                 intercepts, parameters):
        """
        Choice model constructor.

        Args:
            title (str): Title for the problem described by the model object
            choices (list[str]): Labels for the possible choices/alternatives
                in the model
            choice_column (str): Label of the column in the data file which
                will contain the choices for each record
            availability (dict): A dictionary of labels of the columns defining
                the availability of each choice. Keys are the names of each
                choice and the corresponding values are the column labels.
            variables (list[str]): A list of variable names used in utility
                specifications.
            intercepts (dict): A dictionary of intercept variables. The keys
                are the choices to which the intercepts correspond and the
                values are labels for the intercepts. There must be one fewer
                intercept than the number of choices.
            parameters (list[str]): Names of all parameters (except intercepts)
                used in defining the utlity specifications.
        """

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
        Read the model definition from a YAML file.

        Args:
            stream (stream): Data stream of the model definition.

        Returns:
            (ChoiceModel): A choice model object corresponding to the
                definition in the stream.
        """
        return cls(*cls._unpack_yaml(stream))

    @classmethod
    def _unpack_yaml(cls, stream):
        """
        Unpack constructor arguments for model definition from a YAML file.

        Args:
            stream (stream): Data stream of the model definition.

        Returns:
            (tuple): A tuple of the arguments requiredby the constructor.
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

        return (title, choices, choice_column, availability, variables,
                intercepts, parameters)

    @staticmethod
    def _copy_yaml_record(key, yaml_dict, stream):
        """
        Extract the value of a key from the YAML dictionary if it exists, raise
        an exception otherwise.

        Args:
            key (str): The key to search for in the dictionary.
            yaml_dict (dict): The dictionary to search for key.
            stream (stream): The YAML data stream.

        Returns:
            (object): The value corresponding to key in yaml_dict.

        Raises:
            MissingYamlKey: Raised if key is not in yaml_dict.
        """
        if key in yaml_dict:
            return yaml_dict[key]
        else:
            raise MissingYamlKey(key, stream)

    def load_data(self, stream):
        """
        Load data into pandas dataframe.

        Args:
            stream (stream): csv data stream containing model data.
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

    def __init__(self, title, choices, choice_column, availability, variables,
                 intercepts, parameters):
        super().__init__(title, choices, choice_column, availability,
                         variables, intercepts, parameters)

    @classmethod
    def from_yaml(cls, stream):
        return cls(*super()._unpack_yaml(stream))


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
