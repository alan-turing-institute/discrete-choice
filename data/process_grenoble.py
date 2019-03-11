#!/usr/bin/env python3
import numpy as np
import pandas as pd

# Define data file field widths
field_widths = ([4, 2]+[1]*12+[2]+[1]*2+[2]+[1]*2+[2]*2+[1]*2+[4]*2+[2]*2
                + [4, 2]+[4]*2+[5]+[4]*4)
# Data file field names
field_names = ['d1', 'd2', 'd3', 'worklic', 'd5', 'nwlic', 'd7', 'd8', 'd9',
               'cars', 'd11', 'd12', 'd13', 'sex', 'd15', 'household_position',
               'driving_licences', 'occupation', 'd19', 'd20', 'd21', 'mode',
               'd23', 'd24', 'pt_owalk_h', 'pt_dwalk_h', 'origin_zone',
               'destination_zone', 'pt_tot', 'pt_lines', 'pt_owalk',
               'pt_dwalk', 'dist', 'car_time', 'park_orig', 'park_dest',
               'pt_wait']

# Read data
data = pd.read_fwf('./grenoble.dat', widths=field_widths,
                   header=None, names=field_names)

# Clean data
# Drop unecessary columns
data = data.drop(columns=['d1', 'd2', 'd3', 'd5', 'd7', 'd8', 'd9', 'd11',
                          'd12', 'd13', 'd15', 'd19', 'd20', 'd21', 'd23',
                          'd24', 'destination_zone', 'pt_dwalk_h', 'pt_dwalk',
                          'park_orig', 'pt_wait'])
# Set Nan to zero
data = data.applymap(lambda x: 0.0 if np.isnan(x) else x)
# Convert all data to integers
data = data.applymap(lambda x: int(x))


@np.vectorize
def encode_choice(original_code):
    if original_code in [8, 10]:
        return 'public_transport'
    elif original_code == 5:
        return 'car'
    elif original_code in [2, 3, 4]:
        return 'cycle'
    elif original_code == 1:
        return 'walk'
    elif original_code in [6, 7]:
        return 'passenger'
    else:
        return 'other'


# Encode choices
data['mode'] = data['mode'].apply(encode_choice)
# Drop observations with 'other' choice
print('{} bad choices dropped'.format(
    sum(data['mode'] == 'other')))
data = data[data['mode'] != 'other']

# Drop records with >= 5 cars
print('{} observations with >= 5 cars dropped'.format(sum(data['cars'] >= 5)))
data = data[data['cars'] < 5]

# Availability variables
# Availability variables
data['avail_public_transport'] = data['pt_lines'].apply(
        lambda x: 1 if x > 0 else 0)
data['avail_car'] = data[['cars', 'driving_licences']].apply(
        lambda x: 1 if x[0] > 0 and x[1] == 1 else 0, axis=1)
data = data.drop(columns='driving_licences')
data['avail_cycle'] = 1
data['avail_walk'] = data['dist'].apply(lambda x: 1 if x <= 6000 else 0)
data['avail_passenger'] = 1

# Drop records where choice is not available
availability = ['avail_public_transport', 'avail_car', 'avail_cycle',
                'avail_walk', 'avail_passenger']
modes = ['public_transport', 'car', 'cycle', 'walk', 'passenger']
for av, mode in zip(availability, modes):
    bad_records = data[['mode', av]].apply(
        lambda x: True if x[0] == mode and x[1] == 0 else False,
        axis=1)
    n_bad_records = sum(bad_records)
    print('{} records with choice {} but no availability'.format(n_bad_records,
                                                                 mode))
    if n_bad_records > 0:
        print('Dropping bad records')
        data = data[bad_records.apply(lambda x: not x)]

data['head_of_household'] = data['household_position'].apply(
    lambda x: 1 if x == 1 else 0)
data = data.drop(columns='household_position')

# Car competition (something like cars per people)
data['licences'] = data['worklic'] + data['nwlic']
@np.vectorize
def car_comp(cars, licences):
    if licences > 0:
        return min(cars/licences, 1)
    else:
        return 0.


data['car_competition'] = data[['cars', 'licences']].apply(
    lambda x: car_comp(x[0], x[1]), axis=1)
data['car_competition'] = data['car_competition'].apply(float)
data = data.drop(columns=['licences', 'worklic', 'nwlic'])

# Has car variable
data['has_car'] = data['cars'].apply(lambda x: 1 if x >= 1 else 0)
data = data.drop(columns='cars')

# Female variable
data['female'] = data['sex'].apply(lambda x: 1 if x == 2 else 0)
data = data.drop(columns='sex')

# Central zones
central_zones = [1, 2, 3, 5, 6, 8, 10, 11, 12, 14]
data['central_zone'] = data['origin_zone'].apply(
    lambda x: 1 if x in central_zones else 0)
data = data.drop(columns='origin_zone')

# Manual workers
data['manual_worker'] = data['occupation'].apply(
    lambda x: 1 if 60 <= x <= 69 else 0)
data = data.drop(columns='occupation')

# Cycle and walking time in seconds
data['cycle_time'] = data['dist'] * 0.24
data['walk_time'] = data['dist'] * 0.72
data['transit_walk_time'] = data['pt_owalk_h'] * 0.72
data = data.drop(columns='pt_owalk_h')

# Non-linear terms for times between 15 and 30 minutes
@np.vectorize
def non_linear(x):
    threshold = 900
    return min(max(x-threshold, 0), threshold)


data['cycle_non_linear'] = data['cycle_time'].apply(non_linear)
data['walk_non_linear'] = data['walk_time'].apply(non_linear)

# Public transport travel time
data['public_transport_time'] = (data['pt_tot'] - data['pt_owalk']
                                 + 390*data['pt_lines'])
data = data.drop(columns=['pt_tot', 'pt_owalk', 'pt_lines'])

# Public transport flat cost
data['public_transport_cost'] = 75.9

# Driving cost
data['car_cost'] = data['dist']*0.04 + data['park_dest']*3.5
data = data.drop(columns=['dist', 'park_dest'])

data.to_csv('./grenoble.csv', index=False)
