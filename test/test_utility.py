from .context import add_project_path
import choice_model
import pytest

add_project_path()


@pytest.fixture(scope='class')
def simple_utility():
    utility_string = 'c + var1*param1 + param2*var2'
    variables = ['var1', 'var2']
    intercept = 'c'
    parameters = ['param1', 'param2']

    return choice_model.Utility(utility_string, variables, intercept,
                                parameters)


@pytest.fixture(scope='module')
def custom_utility():
    def _custom_utility(utility_string, variables, intercept, parameters):
        return choice_model.Utility(utility_string, variables, intercept,
                                    parameters)
    return _custom_utility


class TestSimpleUtility():
    def test_intercept(self, simple_utility):
        utility = simple_utility
        assert utility.intercept == 'c'

    def test_variables(self, simple_utility):
        utility = simple_utility
        assert utility.variables() == ['var1', 'var2']

    def test_parameters(self, simple_utility):
        utility = simple_utility
        assert utility.parameters() == ['param1', 'param2']


class TestDuplicates():
    def test_duplicate_parameters(self, custom_utility):
        with pytest.raises(choice_model.utility.DuplicateParameters):
            custom_utility(
                utility_string='c + var1*param1 + param1*var2',
                variables=['var1', 'var2'],
                intercept='c',
                parameters=['param1', 'param2']
                )

    def test_duplicate_variables(self, custom_utility):
        with pytest.raises(choice_model.utility.DuplicateVariables):
            custom_utility(
                utility_string='c + var2*param1 + param2*var2',
                variables=['var1', 'var2'],
                intercept='c',
                parameters=['param1', 'param2']
                )


class TestIntercept():
    def test_incorrect_intercept(self, custom_utility):
        with pytest.raises(choice_model.utility.MissingOrIncorrectIntercept):
            custom_utility(
                utility_string='d + var2*param1 + param2*var1',
                variables=['var1', 'var2'],
                intercept='c',
                parameters=['param1', 'param2']
                )

    def test_missing_intercept(self, custom_utility):
        with pytest.raises(choice_model.utility.MissingOrIncorrectIntercept):
            custom_utility(
                utility_string='var2*param1 + param2*var1',
                variables=['var1', 'var2'],
                intercept='c',
                parameters=['param1', 'param2']
                )

    def test_mislocated_intercept(self, custom_utility):
        with pytest.raises(choice_model.utility.MissingOrIncorrectIntercept):
            custom_utility(
                utility_string='var2*param1 + c + param2*var1',
                variables=['var1', 'var2'],
                intercept='c',
                parameters=['param1', 'param2']
                )
