import choice_model
import pandas as pd
import pytest


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

    def test_alternatives(self, synthetic_model):
        assert synthetic_model.alternatives == ['alternative1', 'alternative2']

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

    def test_alternative_independent_variables(self, synthetic_model):
        assert synthetic_model.alternative_independent_variables == []

    def test_alternative_dependent_variables(self, synthetic_model):
        alternative_dependent_variables = {
            'variable1': {'alternative1': 'alternative1_variable1',
                          'alternative2': 'alternative2_variable1'},
            'variable2': {'alternative1': 'alternative1_variable2',
                          'alternative2': 'alternative2_variable2'},
            'variable3': {'alternative1': 'alternative1_variable3',
                          'alternative2': 'alternative2_variable3'}
            }
        assert synthetic_model.alternative_dependent_variables == (
            alternative_dependent_variables)

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

    def test_number_of_alternatives(self, synthetic_model):
        assert synthetic_model.number_of_alternatives() == 2

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
                                       n_observations=5)
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

    def test_alternatives(self, synthetic_data, synthetic_model):
        assert all(synthetic_data['choice'].apply(
            lambda x: x in synthetic_model.alternatives))


@pytest.fixture(scope='module')
def synthetic_data_uniform(synthetic_model):
    data = choice_model.synthetic_data_uniform(model=synthetic_model,
                                               number_of_records=5)
    return data


class TestSyntheticDataUniform():
    def test_data(self, synthetic_data_uniform):
        assert isinstance(synthetic_data_uniform, pd.DataFrame)

    @pytest.mark.parametrize('column', [
        'availability1',
        'availability2',
        ])
    def test_availabilities(self, synthetic_data_uniform, column):
        assert all(synthetic_data_uniform[column] == 1)

    def test_alternatives(self, synthetic_data_uniform, synthetic_model):
        assert all(synthetic_data_uniform['choice'].apply(
            lambda x: x in synthetic_model.alternatives))
