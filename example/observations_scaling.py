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

# Define the example model
model = choice_model.synthetic_model(
    title='Synthetic',
    number_of_alternatives=5,
    number_of_variables=5
    )


def scaling(interface, model, records, repeats, interface_args):
    df = pd.DataFrame(columns=records)
    for number_of_records in records:
        estimation_times = []
        for repeat in range(repeats):
            data = choice_model.synthetic_data(
                model=model,
                number_of_records=number_of_records
                )
            model.load_data(data)
            solver = interface(model, **interface_args)
            solver.estimate()

            estimation_times.append(solver.estimation_time())
        df[number_of_records] = estimation_times
    return df


# Choose which interfaces to compare
if platform.system() == 'Windows':
    interfaces = [choice_model.PylogitInterface,
                  choice_model.AlogitInterface]
    interface_args = {'alogit_path': r'D:\Alo45.exe'}
    records = {'ALOGIT': np.arange(2000, 22000, 2000),
               'pylogit': np.arange(2000, 18000, 2000)}
else:
    interfaces = [choice_model.PylogitInterface]
    interface_args = {}
    records = {'pylogit': np.arange(2000, 18000, 2000)}

# Sample estimation times for each model over a range of observation sizes
estimation_times = {}
for interface in interfaces:
    df = scaling(interface, model, records[interface.name], 10, interface_args)
    estimation_times[interface.name] = df

    with open(
            'scaling_observations_{}.csv'.format(interface.name), 'w'
            ) as csv_file:
        df.to_csv(csv_file,
                  index=False,
                  line_terminator='\n')

fig, ax = plt.subplots()
ax.set_xlabel('number of observations')
ax.set_ylabel('estimation time / s')
for interface in interfaces:
    # Plot the mean estimation times and errors
    results = estimation_times[interface.name]
    ax.errorbar(results.columns, results.mean(),
                yerr=results.sem(), fmt='-x', label=interface.name)

fig.tight_layout()
fig.savefig('scaling_observations.pdf', format='pdf')
