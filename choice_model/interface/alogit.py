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

        self.data_file_path = 'aaa'
        self.alo = self._create_alo_file()

    def _write_alo_file(self):
        # Use first word in title as file prefix
        with open(self.model.title.split(' ')[0] + '.alo', 'w') as alo_file:
            alo_file.write(self.alo)

    def _create_alo_file(self):
        """
        Write the ALOGIT input file
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
            alo += self._add_alo_record('Avail(' + choice + ') =',
                                        model.availability[choice])
        # Define choice column
        alo += self._add_alo_record('choice =', model.choice_column)
        # Write choice dependent variable specification
        for variable, mapping in model.choice_dependent_variables.items():
            # Define the choice dependent variable as an array with size
            # equal to the number of alternatives
            alo += self._add_alo_record(_ALO_COMMAND_ARRAY, variable
                                        + '(alts)')
            # Define the data file column corresponding to each choice
            for choice, column_label in mapping.items():
                alo += self._add_alo_record(variable + '(' + choice
                                            + ') =', column_label)
        # Write utility specifications for each choice
        for choice in model.choices:
            alo += self._add_alo_record('Util(' + choice + ') =',
                                        self._utility_string(choice))
        return alo

    @staticmethod
    def _add_alo_record(command, *args):
        """
        Write a command to the ALOGIT input file
        """
        string = command
        for arg in args:
            string += ' ' + arg
        string += '\n'
        return string

    def _specify_data_file(self, data_file_path):
        """
        Write the line specifying the data file and format to the ALOGIT
        input file.
        """
        # Create space seperated string of column labels
        column_labels = ' '.join(self.model.data.columns)
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
                # utility_string.append(
                #     term.parameter + '*'
                #     + model.choice_dependent_variables[variable][choice]
                #     )
                utility_string.append(term.parameter + '*' + variable + '('
                                      + choice + ')')

            else:
                utility_string.append(term.parameter + '*' + term.variable)

        # Join all terms as a sum
        utility_string = ' + '.join(utility_string)
        return utility_string
