"""
Utility function specification
"""

from collections import namedtuple, Counter

VariableAndParameter = namedtuple('VariableAndParameter',
                                  ['variable', 'parameter'])


class Utility(object):
    """
    Utility class

    Args:
        utility_string (str): A string defining the utility function. The
            utility string has the format "c + p1*v1 + p2*v2 ..." where "c" is
            the intercept, "p" are parameters and "v" are variables. If there
            is an intercept it must be the first term. Parameters
            and variables may occur in any order, but their names must appear
            in the arguments parameters and variables respectively.
        variables (list[str]): A list of variables names that may appear in
            the utility string.
        intercept (str or None): The intercept variable that will appear
            in the utility string or None if there isn't one.
        parameters (list[str]): A list of the parameter names that may
            appear in the utility string.
    """

    def __init__(self, utility_string, variables, intercept, parameters):

        # Split utility string into terms seperated by '+'
        terms = utility_string.split('+')
        # Remove padding spaces
        terms = [term.strip() for term in terms]

        if intercept is None:
            self.intercept = None
        else:
            # Seperate intercept from other terms
            # self.intercept, *terms = terms
            self.intercept, terms = terms[0], terms[1:]
            if self.intercept != intercept:
                raise MissingOrIncorrectIntercept(self.intercept, intercept)

        # Parse terms
        self.terms = []
        for term in terms:
            # identify variable and parameter
            variable_and_parameter = self._split_term(term)
            self.terms.append(
                self._sort_variable_and_parameter(variable_and_parameter,
                                                  variables, parameters)
                )

        # Ensure all variables and parameters appear at most once
        self._check_duplicates()

        self.all_variables = [term.variable for term in self.terms]
        self.all_parameters = [term.parameter for term in self.terms]
        self.term_dict = dict(self.terms)

    @staticmethod
    def _split_term(term):
        """
        Split an individual utility term into its two components.

        Args:
            term (str): The utility term formated as a product of two labels
                _e.g._ "param * var" or "param*var".

        Returns:
            (list[str]): A list of length two containing the two labels (not
                sorted).

        Raises:
            TermNotProduct: Raised when term is not a product of two labels.
        """
        if '*' in term:
            variable_and_parameter = term.split('*')
            variable_and_parameter = [label.strip()
                                      for label in variable_and_parameter]
        else:
            raise TermNotProduct(term)

        if len(variable_and_parameter) != 2:
            raise TermNotProduct(term)

        return variable_and_parameter

    @staticmethod
    def _sort_variable_and_parameter(variable_and_parameter,
                                     variables, parameters):
        """
        Determine which labels in a variable and paramter pair are the
        variable and parameter.

        Args:
            variable_and_parameter (list[str] or tuple[str]): A list of tuple
                of length two containing the labels of the variable and
                parameter.
            variables (list[str]): A list of all variable labels.
            parameters (list[str]): A list of all parameter labels.

        Returns:
            (VariableAndParameter): A named tuple with components 'variable'
                and 'parameter' corresponding to the variable and parameter
                respectively.

        Raises:
            InvalidTermContents: Raised when variable_and_parameter does not
                contain one variable and one parameter.
        """
        a, b = variable_and_parameter

        if a in variables:
            if b in parameters:
                return VariableAndParameter(variable=a, parameter=b)
            else:
                raise InvalidTermContents(a, b)
        elif a in parameters:
            if b in variables:
                return VariableAndParameter(variable=b, parameter=a)
            else:
                raise InvalidTermContents(a, b)
        else:
            raise InvalidTermContents(a, b)

    def variables(self):
        """
        Produce a list of variables in the utility definition

        Returns:
            (list[str]): a list of the variable labels
        """
        return [term.variable for term in self.terms]

    def parameters(self):
        """
        Produce a list of parameters in the utility definition

        Returns:
            (list[str]): a list of the parameter labels
        """
        return [term.parameter for term in self.terms]

    def _check_duplicates(self):
        """
        Ensure there are no duplicate parameters or variables in the utility
        definition
        """
        # check variables
        counter = Counter(self.variables())
        duplicates = [key for key, value in counter.items() if value > 1]
        if duplicates != []:
            raise DuplicateVariables(duplicates)

        # check parameters
        counter = Counter(self.parameters())
        duplicates = [key for key, value in counter.items() if value > 1]
        if duplicates != []:
            raise DuplicateParameters(duplicates)

    def __eq__(self, other):
        if self.intercept == other.intercept and self.terms == other.terms:
            return True
        else:
            return False


class MissingOrIncorrectIntercept(Exception):
    """
    Exception for when the first term of the utility definition is not the
    intercept.
    """
    def __init__(self, intercept, actual):
        super().__init__(
            'The first term of the utility definition "{}" is not the'
            ' specified intercept "{}"'.format(intercept, actual)
            )


class TermNotProduct(Exception):
    """
    Exception for when a term in a utility definition is not a product of two
    labels.
    """
    def __init__(self, term):
        super().__init__(
            'Each non-intercept term in a utility definition must be a'
            ' product of two labels. The offending term is "{}"'.format(term)
            )


class InvalidTermContents(Exception):
    """
    Exception for when the two labels in a utility definition term are not a
    parameter and a variable.
    """
    def __init__(self, a, b):
        super().__init__(
            'Each non-intercept term in a utility definition must be a product'
            ' of a parameter and a variable. The offending labels are "{}"'
            ' and "{}"'.format(a, b)
            )


class DuplicateVariables(Exception):
    """
    Exception for duplicate parameters in a utility definition.
    """
    def __init__(self, variables):
        super().__init__(
            'Variable/s "{}" used more than once in a utility'
            ' definition'.format(variables)
            )


class DuplicateParameters(Exception):
    """
    Exception for duplicate variables in a utility definition.
    """
    def __init__(self, parameters):
        super().__init__(
            'Parameter/s "{}" used more than once in a utility'
            ' definition'.format(parameters)
            )
