#! /usr/bin/env python3

# Add project directory to path
import os
import sys

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(project_dir, 'data/')
sys.path.insert(0, project_dir)

# Import
import choice_model # noqa

# Create model and load data
with open(data_dir+'grenoble.yml') as model_file,\
        open(data_dir+'grenoble.csv') as data_file:
    model = choice_model.MultinomialLogit.from_yaml(model_file)
    model.load_data(data_file)

# Create pylogit interface
pylogit_interface = choice_model.PylogitInterface(model)

# Estimate model using pylogit
pylogit_interface.estimate()

# Print pylogit summary
pylogit_interface.pylogit_model.print_summaries()

# Create alogit interface
alogit_interface = choice_model.AlogitInterface(model)
alogit_interface.data_file_path = 'grenoble.csv'
alogit_interface._write_alo_file()
