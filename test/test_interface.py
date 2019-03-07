from .context import add_project_path, data_dir
import choice_model
import pytest

add_project_path()


@pytest.fixture(scope="class")
def simple_model():
    with open(data_dir+'simple_model.yml', 'r') as yaml_file:
        return choice_model.ChoiceModel.from_yaml(yaml_file)


@pytest.fixture(scope='class')
def simple_multinomial_model():
    with open(data_dir+'simple_model.yml', 'r') as yaml_file:
        return choice_model.MultinomialLogit.from_yaml(yaml_file)


class TestInterface():
    def test_simplemodel(self, simple_model):
        interface = choice_model.Interface(simple_model)
        assert interface.model == simple_model

    def test_multinomial_logit(self, simple_multinomial_model):
        with pytest.raises(TypeError):
            choice_model.Interface(simple_multinomial_model)
