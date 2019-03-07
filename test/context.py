import os
import sys

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(project_dir, 'test/data/')


def add_project_path():
    sys.path.insert(0, project_dir)
