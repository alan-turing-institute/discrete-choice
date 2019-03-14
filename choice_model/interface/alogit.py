"""
ALOGIT interface
"""

from .interface import Interface, requires_estimation
from .. import MultinomialLogit
import numpy as np
import os.path
import subprocess
import textwrap

_ALO_COMMAND_TITLE = '$title '
_ALO_COMMAND_ESTIMATE = '$estimate'
_ALO_COMMAND_COEFFICIENTS = '$coeff'
_ALO_COMMAND_ALTERNATIVES = "$nest root()"
_ALO_COMMAND_ARRAY = "$array"

_MAX_CHARACTER_LENGTH = 10
_MAX_LINE_LENGTH = 77

_CHOICE_COLUMN = 'choice_no'


class AlogitInterface(Interface):
    """
    ALOGIT interface class

    Args:
        model (ChoiceModel): The choice model to create an interface for.
        alogit_path (str): Path to the ALOGIT executable.
        data_file (str, optional, default=None): Path of the file to hold model
            data in the format ALOGIT expects. If 'None' then a prefix is
            created based on the model title and appended with '.csv'
        alo_file (str, optional, default=None): Path of the ALOGIT input (.alo)
            file. If 'None' then a prefix is created based on the model title
            and appended with '.alo'
    """
    _valid_models = [MultinomialLogit]

    def __init__(self, model, alogit_path, data_file=None, alo_file=None):
        super().__init__(model)

        self.alogit_path = os.path.abspath(alogit_path)

        # Define a file prefix for the input and data files
        prefix = self.model.title.split(' ')[0]
        # Define file names
        if data_file is None:
            self.data_file = prefix + '.csv'
        else:
            self.data_file = data_file
        if alo_file is None:
            self.alo_file = prefix + '.alo'
        else:
            self.alo_file = alo_file

        # Create label abbreviations using ALOGIT's maximum character length
        self._create_abbreviations()

        # Create column labels
        column_labels = [self.abbreviate(label)
                         for label in model.data.columns]
        # Replace choice column
        column_labels[
            column_labels.index(self.abbreviate(model.choice_column))
            ] = _CHOICE_COLUMN
        self.column_labels = column_labels

        # Create ALOGIT input file string
        self.alo = self._create_alo_file()

    def _create_abbreviations(self):
        """
        Create abbreviations of variable and parameter names conforming to
        ALOGIT's 10 character limit
        """
        model = self.model

        full = []
        abbreviations = []
        # Abbreviate choice names
        for choice in model.choices:
            full.append(choice)
            abbreviations.append(self._abbreviate(choice))
        # Abbreviate choice names / column label
        choice_column = model.choice_column
        full.append(choice_column)
        abbreviations.append(self._abbreviate(choice_column))
        # Abbreviate availability column labels
        for availability in model.availability.values():
            full.append(availability)
            abbreviations.append(self._abbreviate(availability))
        # Abbreviate variable names / column labels
        all_variables_and_fields = set(model.all_variables()
                                       + model.all_variable_fields())
        for variable in all_variables_and_fields:
            full.append(variable)
            abbreviations.append(self._abbreviate(variable))
        # Abbreviate intercept names
        for intercept in model.intercepts.values():
            full.append(intercept)
            abbreviations.append(self._abbreviate(intercept))
        # Abbreviate parameter names
        for parameter in model.parameters:
            full.append(parameter)
            abbreviations.append(self._abbreviate(parameter))

        # Handle duplicates due to truncation (only up to ten duplicates)
        for abbreviation in abbreviations[:]:
            # Count number of duplicates of abbreviation
            duplicate_count = abbreviations.count(abbreviation)
            if duplicate_count > 1:
                # Replace each occurance usings numbers 1--duplicate_count
                for occurance in range(1, duplicate_count+1):
                    index = abbreviations.index(abbreviation)
                    abbreviations[index] = (
                        abbreviation[:-1] + str(occurance)
                        )

        self.abbreviation = dict(zip(full, abbreviations))
        self.elongation = dict(zip(abbreviations, full))

    @staticmethod
    def _abbreviate(string):
        """
        'Abbreviate' a string by truncating it.
        """
        return string[:_MAX_CHARACTER_LENGTH]

    def abbreviate(self, string):
        """
        Abbreviate a string if its abbreviation has been defined.

        Args:
            string (str): The string to attempt to abbreviate.

        Returns:
            (str): The abbreviation of string if it has been defined by the
            interface, string otherwise.
        """
        if string in self.abbreviation:
            return self.abbreviation[string]
        else:
            return string

    def elongate(self, string):
        """
        Produce the long form of an abbreviation defined by the interface.

        Args:
            string (str): The string to elongate.

        Returns:
            (str): The long form of string as defined by the model.

        Raises:
            KeyError: If string is not an abbreviation defined by the
            interface.
        """
        return self.elongation[string]

    def _write_alo_file(self):
        """
        Write ALOGIT input file string to a file
        """
        # Use first word in title as file prefix
        with open(self.alo_file, 'w') as alo_file:
            for line in self.alo:
                alo_file.write(line + '\n')

    def _create_alo_file(self):
        """
        Create ALOGIT input file string
        """
        model = self.model
        alo = []
        # Write title
        alo += self._alo_record(_ALO_COMMAND_TITLE, model.title)
        # Estimate instruction
        alo += self._alo_record(_ALO_COMMAND_ESTIMATE)
        # Write coefficients (parameters and intercepts)
        alo += self._alo_record(
            _ALO_COMMAND_COEFFICIENTS,
            *model.parameters + list(model.intercepts.values())
            )
        # Write alternatives (choices)
        alo += self._alo_record(_ALO_COMMAND_ALTERNATIVES, *model.choices)
        # Write data file specification
        alo += self._specify_data_file()
        # Write availability columns
        for choice in model.choices:
            alo += self._alo_record(self._array_record('Avail', choice),
                                    model.availability[choice])
        # Define choices
        alo += self._define_choices()
        # Write choice dependent variable specification
        for variable, mapping in model.choice_dependent_variables.items():
            # Define the choice dependent variable as an array with size
            # equal to the number of alternatives
            alo += self._alo_record(_ALO_COMMAND_ARRAY,
                                    self._array(variable, 'alts'))
            # Define the data file column corresponding to each choice
            for choice, column_label in mapping.items():
                alo += self._alo_record(
                    self._array_record(variable, choice), column_label)
        # Write utility specifications for each choice
        for choice in model.choices:
            alo += self._alo_record(self._array_record('Util', choice),
                                    self._utility_string(choice))
        return alo

    def _alo_record(self, command, *args):
        """
        Write a record to the ALOGIT input file
        """
        string = command
        for arg in args:
            string += ' ' + self.abbreviate(arg)
        return textwrap.wrap(string, width=_MAX_LINE_LENGTH,
                             break_long_words=False)

    def _array(self, array, argument):
        """
        Format an array in the form "array(argument)"
        """
        array = self.abbreviate(array)
        argument = self.abbreviate(argument)
        return array + '(' + argument + ')'

    def _array_record(self, array, argument):
        """
        Format an array record in the form "array(argument) ="
        """
        return self._array(array, argument) + ' ='

    def _specify_data_file(self):
        """
        Write the line specifying the data file and format to the ALOGIT
        input file.
        """
        # Create space seperated string of column labels
        column_labels = ' '.join(self.column_labels)
        string = 'file (name=' + self.data_file + ') ' + column_labels
        return textwrap.wrap(string, width=_MAX_LINE_LENGTH,
                             break_long_words=False)

    def _define_choices(self):
        """
        Create a record to explain the numeric encoding of choices
        """
        model = self.model
        string = 'choice=recode(' + _CHOICE_COLUMN + ' ' + ', '.join(
            [self.abbreviate(choice) for choice in model.choices]) + ')'
        return textwrap.wrap(string, width=_MAX_LINE_LENGTH,
                             break_long_words=False)

    def _utility_string(self, choice):
        """
        Construct an ALOGIT style utility string
        """
        model = self.model
        utility = self.model.specification[choice]
        choice_dependent_variables = model.choice_dependent_variables.keys()

        # Intercept term
        if utility.intercept is not None:
            utility_string = [utility.intercept]
        else:
            utility_string = []

        # parameter * variable terms
        for term in utility.terms:
            variable = term.variable
            # Format choice dependent variables
            if variable in choice_dependent_variables:
                utility_string.append(
                    self.abbreviate(term.parameter) + '*'
                    + self.abbreviate(variable) + '('
                    + self.abbreviate(choice) + ')'
                    )
            else:
                utility_string.append(
                    self.abbreviate(term.parameter) + '*'
                    + self.abbreviate(term.variable)
                    )

        # Join all terms as a sum
        utility_string = ' + '.join(utility_string)
        return utility_string

    def _write_data_file(self):
        """
        Write the data in the format defined by the ALOGIT input file
        """
        model = self.model

        # Encode choices as numbers in new dataframe column
        number_of_choices = model.number_of_choices()
        choice_encoding = dict(
            zip(model.choices, np.arange(number_of_choices, dtype=float)+1))
        model.data[_CHOICE_COLUMN] = model.data[model.choice_column].apply(
            lambda x: choice_encoding[x])

        # Produce list of column labels replacing old choice column with the
        # new encoded choice column
        column_labels = list(model.data.columns)[:-1]
        column_labels[
            column_labels.index(model.choice_column)
            ] = _CHOICE_COLUMN

        # Write data file
        with open(self.data_file, 'w') as data_file:
            model.data.to_csv(data_file, header=False, index=False,
                              line_terminator='\n',
                              columns=column_labels)

        # Drop encoded choice column
        model.data.drop(columns=_CHOICE_COLUMN, inplace=True)

    def estimate(self):
        """
        Estimate the parameters of the choice model using ALOGIT.
        """
        # Write the input and data files
        self._write_alo_file()
        self._write_data_file()

        alo_path = os.path.abspath(self.alo_file)

        # Call ALOGIT
        process = subprocess.run([self.alogit_path, alo_path],
                                 capture_output=True)

        # Set estimated flag if ALOGIT ran successfully
        if process.returncode == 0:
            self._estimated = True

        self.process = process

    @requires_estimation
    def display_results(self):
        """
        Print the results of estimation
        """
        process = self.process
        if process.returncode != 0:
            print('ALOGIT returned non-zero return code')
            print(process.stderr.decode('utf-8'))
        else:
            print(process.stdout.decode('utf-8'))
