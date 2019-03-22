import pytest
import os


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
