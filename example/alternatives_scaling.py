#! /usr/bin/env python3
import choice_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import platform


# Define the example model
def create_model(number_of_alternatives):
    model = choice_model.synthetic_model(
        title='Synthetic',
        number_of_alternatives=number_of_alternatives,
        number_of_variables=5
        )
    return model


def scaling(interface, model, n_observations, repeats, interface_args):
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
    return estimation_times


if platform.system() == 'Windows':
    interfaces = [choice_model.PylogitInterface,
                  choice_model.AlogitInterface]
    interface_args = {'alogit_path': r'D:\Alo45.exe'}
else:
    interfaces = [choice_model.PylogitInterface]
    interface_args = {}

number_of_alternatives = np.arange(5, 55, 5)
n_observations = 5000
repeats = 10

estimation_times = {}
for interface in interfaces:
    print('interface: {}'.format(interface.name))
    df = pd.DataFrame(columns=number_of_alternatives)
    for n in number_of_alternatives:
        print('number of alternatives: {}'.format(n))
        model = create_model(n)

        df[n] = scaling(
            interface,
            model,
            n_observations,
            repeats,
            interface_args
            )
    estimation_times[interface.name] = df

    with open(
            'scaling_alternatives_{}.csv'.format(interface.name), 'w'
            ) as csv_file:
        df.to_csv(csv_file,
                  index=False,
                  line_terminator='\n')

fig, ax = plt.subplots()
ax.set_xlabel('number of alternatives')
ax.set_ylabel('estimation time / s')
for interface in interfaces:
    # Plot the mean estimation times and errors
    results = estimation_times[interface.name]
    ax.errorbar(results.columns, results.mean(),
                yerr=results.sem(), fmt='-o', label=interface.name)

fig.legend()
fig.tight_layout()
fig.savefig('scaling_alternatives.pdf', format='pdf')
