import choice_model
import pandas
import pytest


@pytest.fixture(scope="class")
def simple_model():
    with open(data_dir+'simple_model.yml', 'r') as yaml_file:
        return choice_model.ChoiceModel.from_yaml(yaml_file)


@pytest.fixture(scope="class")
def simple_model_with_data():
    with open(data_dir+'simple_model.yml', 'r') as yaml_file,\
            open(data_dir+'simple.csv', 'r') as data_file:
        model = choice_model.ChoiceModel.from_yaml(yaml_file)
        model.load_data(data_file)
        return model


class TestChoiceModel():
    def test_model_title(self, simple_model):
        model = simple_model
        assert model.title == 'Simple model'

    def test_model_data(self, simple_model_with_data):
        model = simple_model_with_data
        assert all(model.data == pandas.read_csv(data_dir+'simple.csv'))

    def test_model_choices(self, simple_model):
        model = simple_model
        assert model.choices == ['choice1', 'choice2']

    def test_model_choice_column(self, simple_model):
        model = simple_model
        assert model.choice_column == 'alternative'

    def test_model_variables(self, simple_model):
        model = simple_model
        assert model.all_variables() == ['var1', 'var2', 'var3']

    def test_model_variables_fields(self, simple_model):
        model = simple_model
        assert model.all_variable_fields() == ['var1', 'var2', 'choice1_var3',
                                               'choice2_var3']

    def test_model_intercepts(self, simple_model):
        model = simple_model
        assert model.intercepts == {'choice1': 'cchoice1'}

    def test_model_parameters(self, simple_model):
        model = simple_model
        assert model.parameters == ['p1', 'p2', 'p3']

    def test_number_of_choices(self, simple_model):
        model = simple_model
        assert model.number_of_choices() == 2

    def test_number_of_parameters(self, simple_model):
        model = simple_model
        assert model.number_of_parameters() == 4

    def test_number_of_parameters_excluding_intercepts(self, simple_model):
        model = simple_model
        assert model.number_of_parameters(include_intercepts=False) == 3


def test_missing_yaml_key():
    with pytest.raises(choice_model.model.MissingYamlKey):
        with open(data_dir+'missing_title.yml', 'r') as yaml_file:
            choice_model.ChoiceModel.from_yaml(yaml_file)


def test_undefined_availability():
    with pytest.raises(choice_model.model.UndefinedAvailability):
        with open(data_dir+'undefined_availability.yml', 'r') as yaml_file:
            choice_model.ChoiceModel.from_yaml(yaml_file)


def test_incorrect_intercepts():
    with pytest.raises(choice_model.model.IncorrectNumberOfIntercepts):
        with open(data_dir+'incorrect_intercepts.yml', 'r') as yaml_file:
            choice_model.ChoiceModel.from_yaml(yaml_file)


def test_missing_field():
    with pytest.raises(choice_model.model.MissingField):
        with open(data_dir+'simple_model.yml', 'r') as yaml_file,\
                open(data_dir+'missing_field.csv', 'r') as data_file:
            model = choice_model.ChoiceModel.from_yaml(yaml_file)
            model.load_data(data_file)


@pytest.fixture(scope='class')
def simple_multinomial_model():
    with open(data_dir+'simple_model.yml', 'r') as yaml_file:
        return choice_model.MultinomialLogit.from_yaml(yaml_file)


class TestMultinomialLogit():
    utility_string1 = 'cchoice1 + p1* var1 + p3*var3'
    utility_string2 = 'p2* var2 + p3*var3'
    variables = ['var1', 'var2', 'var3']
    intercept = 'cchoice1'
    parameters = ['p1', 'p2', 'p3']
    u1 = choice_model.Utility(utility_string1, variables, intercept,
                              parameters)
    u2 = choice_model.Utility(utility_string2, variables, None, parameters)

    @pytest.mark.parametrize('choice,utility', [
        ('choice1', u1),
        ('choice2', u2)
        ])
    def test_specification(self, simple_multinomial_model,
                           choice, utility):
        model = simple_multinomial_model
        assert model.specification[choice] == utility
