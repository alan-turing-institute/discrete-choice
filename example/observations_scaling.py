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
import platform # noqa

model = choice_model.synthetic_model(
    title='Synthetic',
    number_of_alternatives=5,
    number_of_variables=5
    )


def scaling(interface, model, records):
    estimation_times = []
    for number_of_records in records:
        data = choice_model.synthetic_data(
            model=model,
            number_of_records=number_of_records
            )
        model.load_data(data)
        solver = interface(model)
        solver.estimate()
        estimation_times.append(solver.estimation_time())

    return estimation_times


if platform.system() == 'Windows':
    interfaces = [choice_model.PylogitInterface,
                  choice_model.AlogitInterface]
else:
    interfaces = [choice_model.PylogitInterface]

records = np.arange(1000, 22000, 2000)
estimation_times = []
for interface in interfaces:
    estimation_times.append(scaling(interface, model, records))

fig, ax = plt.subfigs()
ax.set_xlabel('number of observations')
ax.set_ylabel('estimation time / s')
for interface, times in zip(interfaces, estimation_times):
    print(interface.name)
    print(list(zip(records, times)))
    plt.plot(records, times, '-x', label=interface.name)

fig.tightlayout()
fig.savefig('scaling_observations.pdf', format='pdf')
