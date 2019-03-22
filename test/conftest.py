import choice_model
import os
import pytest


@pytest.fixture(scope='session')
def project_dir():
    project_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
        )
    return project_dir


@pytest.fixture(scope='session')
def data_dir(project_dir):
    data_dir = os.path.join(project_dir, 'test/data/')
    return data_dir


@pytest.fixture(scope='session')
def main_data_dir(project_dir):
    data_dir = os.path.join(project_dir, 'data/')
    return data_dir


@pytest.fixture(scope='session')
def simple_model(data_dir):
    with open(data_dir+'simple_model.yml', 'r') as yaml_file:
        return choice_model.ChoiceModel.from_yaml(yaml_file)


@pytest.fixture(scope='session')
def simple_multinomial_model(data_dir):
    with open(data_dir+'simple_model.yml', 'r') as yaml_file:
        return choice_model.MultinomialLogit.from_yaml(yaml_file)


@pytest.fixture(scope="class")
def simple_model_with_data(simple_model, data_dir):
    model = simple_model
    with open(data_dir+'simple.csv', 'r') as data_file:
        model.load_data(data_file)
    return model


@pytest.fixture(scope='session')
def simple_multinomial_model_with_data(simple_multinomial_model, data_dir):
    model = simple_multinomial_model
    with open(data_dir+'simple.csv', 'r') as data_file:
        model.load_data(data_file)
    return model
