"""
Choice model definitions.
"""

from io import IOBase
import pandas as pd
from .utility import Utility
import yaml


class ChoiceModel(object):
    """
    Parent class for choice models.
    """

    def __init__(self, title, alternatives, choice_column, availability,
                 alternative_independent_variables,
                 alternative_dependent_variables, intercepts, parameters):
        """
        Choice model constructor.

        Args:
            title (str): Title for the problem described by the model object
            alternatives (list[str]): Labels for the possible alternatives
                in the model
            choice_column (str): Label of the column in the data file which
                will contain the alternatives for each record
            availability (dict): A dictionary of labels of the columns defining
                the availability of each choice. Keys are the names of each
                choice and the corresponding values are the column labels.
            alternative_independent_variables (list[str]): A list of variable
                names used in utility specifications that do not vary with
                choice.
            alternative_dependent_variables (dict): A dictionary defining
                variables that vary with choice. The keys of the dictionary are
                the names of the variables used in the utility specifications.
                The values are themselves dictionaries, with the key specifying
                the label of the choice and the value being the label of the
                field in the data file.  For example:
                    {'travel_time': {'car': 'car_time',
                                     'bus': 'bus_time'},
                     'cost': {'car': 'cost_car',
                              'bus': 'cost_bus'},
                    }
            intercepts (dict): A dictionary of intercept variables. The keys
                are the alternatives to which the intercepts correspond and the
                values are labels for the intercepts. There must be one fewer
                intercept than the number of alternatives.
            parameters (list[str]): Names of all parameters (except intercepts)
                used in defining the utlity specifications.
        """

        self.title = title
        self.alternatives = alternatives
        self.choice_column = choice_column
        self.availability = availability
        self.alternative_independent_variables = (
            alternative_independent_variables
            )
        self.alternative_dependent_variables = alternative_dependent_variables
        self.intercepts = intercepts
        self.parameters = parameters
        self.data = None

        # Ensure all alternatives have an availability variable
        self._check_availability()

        # Ensure that there are enough intercepts (one fewer than the number of
        # alternatives)
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
        model_dict = yaml.load(stream, Loader=yaml.FullLoader)
        return cls(*cls._unpack_yaml(model_dict))

    @classmethod
    def _unpack_yaml(cls, model_dict):
        """
        Unpack constructor arguments for model definition a YAML dictionary.

        Args:
            model_dict (dict): Dictionary of model definition.

        Returns:
            (tuple): A tuple of the arguments requiredby the constructor.
        """
        title = cls._copy_yaml_record('title', model_dict)
        alternatives = cls._copy_yaml_record('alternatives', model_dict)
        choice_column = cls._copy_yaml_record('choice_column', model_dict)
        availability = cls._copy_yaml_record('availability', model_dict)
        alternative_independent_variables = cls._copy_yaml_record(
            'alternative_independent_variables', model_dict)
        alternative_dependent_variables = cls._copy_yaml_record(
            'alternative_dependent_variables', model_dict)
        intercepts = cls._copy_yaml_record('intercepts', model_dict)
        parameters = cls._copy_yaml_record('parameters', model_dict)

        return (title, alternatives, choice_column, availability,
                alternative_independent_variables,
                alternative_dependent_variables, intercepts, parameters)

    @staticmethod
    def _copy_yaml_record(key, yaml_dict):
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
            raise MissingYamlKey(key)

    def load_data(self, data_or_file):
        """
        Load data into pandas dataframe.

        Args:
            data_or_file (DataFrame or FileLike): Pandas dataframe or file
                object containing the data to load into the model.
        """
        if isinstance(data_or_file, pd.DataFrame):
            self.data = data_or_file
        elif isinstance(data_or_file, IOBase):
            self.data = pd.read_csv(data_or_file)
        else:
            raise TypeError(
                'The argument to load_data must be a pandas dataframe or a '
                'file-like object'
                )

        # Ensure that all required fields are defined in the dataframe
        self._check_fields(data_or_file)

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
        for variable in self.all_variable_fields():
            if variable not in dataframe_columns:
                raise MissingField(variable, stream)

    def _check_availability(self):
        for choice in self.alternatives:
            if choice not in self.availability:
                raise UndefinedAvailability(choice)

    def _check_intercepts(self):
        n_intercepts = len(self.intercepts)
        n_required = len(self.alternatives) - 1
        if n_intercepts != n_required:
            raise IncorrectNumberOfIntercepts(n_intercepts, n_required)

    def all_variables(self):
        """
        Produce a list of all variables in the model, both choice dependent and
        independent.
        """
        alternative_independent = self.alternative_independent_variables
        alternative_dependent = list(
            self.alternative_dependent_variables.keys()
            )
        return alternative_independent + alternative_dependent

    def number_of_variables(self):
        """
        Return the total number of variables (both alternative dependent and
        independent).
        """
        return (len(self.alternative_independent_variables)
                + len(self.alternative_dependent_variables))

    def alternative_dependent_variable_fields(self):
        """
        Produce a list of all expected fields in the data file corresponding to
        choice dependent variables.
        """
        alternative_dependent = [
            label for variable in self.alternative_dependent_variables.values()
            for label in variable.values()
            ]
        return alternative_dependent

    def all_variable_fields(self):
        """
        Produce a list of all expected fields in the data file corresponding to
        choice dependent or independent variables.
        """
        alternative_independent = self.alternative_independent_variables
        alternative_dependent = self.alternative_dependent_variable_fields()
        return alternative_independent + alternative_dependent

    def availability_fields(self):
        """
        Produce a list of all availability fields expected in the data.
        """
        return list(self.availability.values())

    def number_of_alternatives(self):
        """
        Determine the number of alternatives in the model
        """
        return len(self.alternatives)

    def number_of_parameters(self, include_intercepts=True):
        """
        Determine the number of parameters/coefficients in the model

        Args:
            including_intercepts (bool, default=True): if True include the
                number of intercepts in the count.
        """
        number_of_parameters = len(self.parameters)
        if include_intercepts:
            number_of_parameters += len(self.intercepts)
        return number_of_parameters


class MultinomialLogit(ChoiceModel):
    """
    Multinomial logit choice model class.
    """
    def __init__(self, title, alternatives, choice_column, availability,
                 alternative_independent_variables,
                 alternative_dependent_variables, intercepts, parameters,
                 specification):
        super().__init__(title, alternatives, choice_column, availability,
                         alternative_independent_variables,
                         alternative_dependent_variables, intercepts,
                         parameters)

        # Create utility definitions
        self.specification = {}
        for choice in self.alternatives:
            if choice in self.intercepts:
                intercept = self.intercepts[choice]
            else:
                intercept = None
            self.specification[choice] = Utility(specification[choice],
                                                 self.all_variables(),
                                                 intercept,
                                                 self.parameters)

    @classmethod
    def from_yaml(cls, stream):
        model_dict = yaml.load(stream, Loader=yaml.FullLoader)
        specification = cls._copy_yaml_record('specification', model_dict)

        return cls(*super()._unpack_yaml(model_dict), specification)


class MissingYamlKey(Exception):
    """
    Exception for missing, required YAML keys.
    """
    def __init__(self, key):
        super().__init__(
            'Required key "{}" missing'.format(key)
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
