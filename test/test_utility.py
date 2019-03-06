from .context import add_project_path
import choice_model
import pytest

add_project_path()


@pytest.fixture(scope="class")
def simple_utility():
    utility_string = 'c + var1*param1 + param2*var2'
    variables = ['var1', 'var2']
    intercept = 'c'
    parameters = ['param1', 'param2']

    return choice_model.Utility(utility_string, variables, intercept,
                                parameters)


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
