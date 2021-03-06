#! /usr/bin/env python3
import choice_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import platform

# Define the example model
model = choice_model.synthetic_model(
    title='Synthetic',
    number_of_alternatives=5,
    number_of_variables=5
    )


def scaling(interface, model, records, repeats, interface_args):
    df = pd.DataFrame(columns=records)
    for n_observations in records:
        print('Number of observations: {}'.format(n_observations))
        estimation_times = []
        for repeat in range(repeats):
            print('\trepeat: {}'.format(repeat))
            data = choice_model.synthetic_data(
                model=model,
                n_observations=n_observations
                )
            model.load_data(data)
            solver = interface(model, **interface_args)
            solver.estimate()

            estimation_times.append(solver.estimation_time())
        df[n_observations] = estimation_times
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
    print('Interface: {}'.format(interface.name))
    # Plot the mean estimation times and errors
    results = estimation_times[interface.name]
    ax.errorbar(results.columns, results.mean(),
                yerr=results.sem(), fmt='-o', label=interface.name)

fig.legend()
fig.tight_layout()
fig.savefig('scaling_observations.pdf', format='pdf')
