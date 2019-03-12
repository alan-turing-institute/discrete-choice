"""
ALOGIT interface
"""

from . import Interface
from .. import MultinomialLogit

_ALO_COMMAND_TITLE = '$title '
_ALO_COMMAND_ESTIMATE = '$estimate'
_ALO_COMMAND_COEFFICIENTS = '$coeff'
_ALO_COMMAND_ALTERNATIVES = "$nest root()"
_ALO_COMMAND_ARRAY = "$array"

_MAX_CHARACTER_LENGTH = 10


class AlogitInterface(Interface):
    """
    ALOGIT interface class

    Args:
        model (ChoiceModel): The choice model to create an interface for.
        alogit_path (str, optional): Path to the alogit executable.
    """
    _valid_models = [MultinomialLogit]

    def __init__(self, model, alogit_path=None):
        super().__init__(model)

        self._create_abbreviations()
        self.data_file_path = 'aaa'
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
            duplicate_count = abbreviations.count(abbreviation)
            if duplicate_count > 1:
                for occurance in range(1, duplicate_count+1):
                    print(occurance, abbreviation)
                    index = abbreviations.index(abbreviation)
                    abbreviations[index] = (
                        abbreviation[:-1] + str(occurance)
                        )

        self.abbreviation = dict(zip(full, abbreviations))
        self.elongation = dict(zip(abbreviations, full))

    @staticmethod
    def _abbreviate(string):
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
        with open(self.model.title.split(' ')[0] + '.alo', 'w') as alo_file:
            alo_file.write(self.alo)

    def _create_alo_file(self):
        """
        Create ALOGIT input file string
        """
        model = self.model
        alo = ''
        # Write title
        alo += self._add_alo_record(_ALO_COMMAND_TITLE, model.title)
        # Estimate instruction
        alo += self._add_alo_record(_ALO_COMMAND_ESTIMATE)
        # Write coefficients (parameters)
        alo += self._add_alo_record(_ALO_COMMAND_COEFFICIENTS,
                                    *model.parameters)
        # Write alternatives (choices)
        alo += self._add_alo_record(_ALO_COMMAND_ALTERNATIVES,
                                    *model.choices)
        # Write data file specification
        alo += self._specify_data_file(self.data_file_path)
        # Write availability columns
        for choice in model.choices:
            alo += self._add_alo_record(self._array_record('Avail', choice),
                                        model.availability[choice])
        # Define choice column
        alo += self._add_alo_record('choice =', model.choice_column)
        # Write choice dependent variable specification
        for variable, mapping in model.choice_dependent_variables.items():
            # Define the choice dependent variable as an array with size
            # equal to the number of alternatives
            alo += self._add_alo_record(_ALO_COMMAND_ARRAY,
                                        self._array(variable, 'alts'))
            # Define the data file column corresponding to each choice
            for choice, column_label in mapping.items():
                alo += self._add_alo_record(
                    self._array_record(variable, choice), column_label)
        # Write utility specifications for each choice
        for choice in model.choices:
            alo += self._add_alo_record(self._array_record('Util', choice),
                                        self._utility_string(choice))
        return alo

    def _add_alo_record(self, command, *args):
        """
        Write a record to the ALOGIT input file
        """
        string = command
        for arg in args:
            string += ' ' + self.abbreviate(arg)
        string += '\n'
        return string

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

    def _specify_data_file(self, data_file_path):
        """
        Write the line specifying the data file and format to the ALOGIT
        input file.
        """
        # Create space seperated string of column labels
        column_labels = [self.abbreviate(label)
                         for label in self.model.data.columns]
        column_labels = ' '.join(column_labels)
        string = 'file (name=' + data_file_path + ') ' + column_labels + '\n'
        return string

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
