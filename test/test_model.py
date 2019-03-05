from .context import add_project_path, data_dir
import choice_model
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
