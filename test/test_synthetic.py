from .context import add_project_path
import choice_model
import pandas as pd
import pytest
add_project_path()


@pytest.fixture(scope='module')
def synthetic_model():
    model = choice_model.synthetic_model(
        title='Example Title',
        number_of_alternatives=2,
        number_of_variables=3
        )
    return model


class TestSyntheticmodel():
    def test_title(self, synthetic_model):
        assert synthetic_model.title == 'Example Title'

    def test_choices(self, synthetic_model):
        assert synthetic_model.choices == ['alternative1', 'alternative2']

    def test_variables(self, synthetic_model):
        assert synthetic_model.all_variables() == ['variable1', 'variable2',
                                                   'variable3']

    def test_choice_column(self, synthetic_model):
        assert synthetic_model.choice_column == 'choice'

    def test_availability(self, synthetic_model):
        assert synthetic_model.availability == {
            'alternative1': 'availability1',
            'alternative2': 'availability2'
            }

    def test_choice_independent_variables(self, synthetic_model):
        assert synthetic_model.choice_independent_variables == []

    def test_choice_dependent_variables(self, synthetic_model):
        choice_dependent_variables = {
            'variable1': {'alternative1': 'alternative1_variable1',
                          'alternative2': 'alternative2_variable1'},
            'variable2': {'alternative1': 'alternative1_variable2',
                          'alternative2': 'alternative2_variable2'},
            'variable3': {'alternative1': 'alternative1_variable3',
                          'alternative2': 'alternative2_variable3'}
            }
        assert synthetic_model.choice_dependent_variables == (
            choice_dependent_variables)

    def test_intercepts(self, synthetic_model):
        assert synthetic_model.intercepts == {'alternative1': 'c1'}

    def test_parameters(self, synthetic_model):
        assert synthetic_model.parameters == ['parameter1', 'parameter2',
                                              'parameter3']

    def test_variable_fields(self, synthetic_model):
        variable_fields = ['alternative1_variable1', 'alternative2_variable1',
                           'alternative1_variable2', 'alternative2_variable2',
                           'alternative1_variable3', 'alternative2_variable3']
        assert synthetic_model.all_variable_fields() == variable_fields

    def test_number_of_choices(self, synthetic_model):
        assert synthetic_model.number_of_choices() == 2

    @pytest.mark.parametrize('include_intercepts,result', [
        (True, 4),
        (False, 3)
        ])
    def test_number_of_parameters(self, synthetic_model, include_intercepts,
                                  result):
        assert synthetic_model.number_of_parameters(
            include_intercepts) == result


@pytest.fixture(scope='module')
def synthetic_data(synthetic_model):
    data = choice_model.synthetic_data(model=synthetic_model,
                                       number_of_records=5)
    return data


class TestSyntheticData():
    def test_data(self, synthetic_data):
        assert isinstance(synthetic_data, pd.DataFrame)

    @pytest.mark.parametrize('column', [
        'availability1',
        'availability2',
        ])
    def test_availabilities(self, synthetic_data, column):
        assert all(synthetic_data[column] == 1)

    def test_choices(self, synthetic_data, synthetic_model):
        assert all(synthetic_data['choice'].apply(
            lambda x: x in synthetic_model.choices))
