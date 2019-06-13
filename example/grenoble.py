#! /usr/bin/env python3
import choice_model
import os
import platform

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))

# Create model and load data
with open(data_dir+'/grenoble.yml') as model_file,\
        open(data_dir+'/grenoble.csv') as data_file:
    model = choice_model.MultinomialLogit.from_yaml(model_file)
    model.load_data(data_file)

if platform.system() == 'Windows':
    interfaces = [choice_model.AlogitInterface,
                  choice_model.BiogemeInterface,
                  choice_model.PylogitInterface
                  ]
    interface_args = dict(alogit_path=r'D:\Alo45.exe')
else:
    interfaces = [choice_model.BiogemeInterface,
                  choice_model.PylogitInterface
                  ]
    interface_args = {}

for interface in interfaces:
    solver = interface(model, **interface_args)
    solver.estimate()
    print('\n')
    print(solver.name)
    print('\tlog-likelihood: {:.3f}'.format(solver.final_log_likelihood()))
    print('\tparameters:')
    for parameter, value in sorted(solver.parameters().items()):
        print('\t\t{:20s} {:0< 6.3g}'.format(parameter, value), sep='')
