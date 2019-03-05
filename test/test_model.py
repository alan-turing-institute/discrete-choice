from .context import add_project_path, data_dir
import choice_model
import pandas
import pytest

add_project_path()


@pytest.fixture(scope="class")
def simple_model():
    print(data_dir)
    with open(data_dir+'simple_model.yml') as yaml_file:
        return choice_model.ChoiceModel.from_yaml(yaml_file)


class TestChoiceModel():
    def test_model_title(self, simple_model):
        model = simple_model
        assert model.title == 'Simple model'

    def test_model_data(self, simple_model):
        model = simple_model
        assert all(model.data == pandas.read_csv(data_dir+'simple.csv'))

    def test_model_choices(self, simple_model):
        model = simple_model
        assert model.choices == ['choice1', 'choice2']

    def test_model_choice_column(self, simple_model):
        model = simple_model
        assert model.choice_column == 'alternative'

    def test_model_variables(self, simple_model):
        model = simple_model
        assert model.variables == ['var1', 'var2', 'var3', 'var4']

    def test_model_intercepts(self, simple_model):
        model = simple_model
        assert model.intercepts == {'var1': 'cchoice1'}

    def test_model_parameters(self, simple_model):
        model = simple_model
        assert model.parameters == ['p1', 'p2']
