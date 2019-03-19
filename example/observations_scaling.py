#! /usr/bin/env python3

# Add project directory to path
import os
import sys

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(project_dir, 'data/')
sys.path.insert(0, project_dir)

# Import
import choice_model # noqa
import matplotlib.pyplot as plt # noqa
import numpy as np # noqa
import pandas as pd # noqa
import platform # noqa

model = choice_model.synthetic_model(
    title='Synthetic',
    number_of_alternatives=5,
    number_of_variables=5
    )


def scaling(interface, model, records, repeats, interface_args):
    estimation_times = []
    for number_of_records in records:
        average = 0.
        for repeat in range(repeats):
            data = choice_model.synthetic_data(
                model=model,
                number_of_records=number_of_records
                )
            model.load_data(data)
            solver = interface(model, **interface_args)
            solver.estimate()

            average += solver.estimation_time()
            del data
            del solver
        estimation_times.append(average/repeats)

    return estimation_times


if platform.system() == 'Windows':
    interfaces = [choice_model.PylogitInterface,
                  choice_model.AlogitInterface]
    interface_args = {'alogit_path': r'D:\Alo45.exe'}
else:
    interfaces = [choice_model.PylogitInterface]
    interface_args = {}

records = np.arange(2000, 22000, 2000)
df = pd.DataFrame()
df['number of observations'] = records
for interface in interfaces:
    df[interface.name] = scaling(interface, model, records, 5, interface_args)

with open('scaling_observations.csv', 'w') as csv_file:
    df.to_csv(csv_file,
              index=False,
              line_terminator='\n')

fig, ax = plt.subplots()
ax.set_xlabel('number of observations')
ax.set_ylabel('estimation time / s')
for interface in interfaces:
    plt.plot(records, df[interface.name], '-x', label=interface.name)

fig.tight_layout()
fig.savefig('scaling_observations.pdf', format='pdf')
