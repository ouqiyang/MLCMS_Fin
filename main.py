import src.utils as utils

bottleneck_data_path = '/Users/ardacaliskan/Documents/lab/P08-pml4effpred/data/Pedestrian_Trajectories/Bottleneck_Data'
corridor_data_path = '/Users/ardacaliskan/Documents/lab/P08-pml4effpred/data/Pedestrian_Trajectories/Corridor_Data'

# Processing the bottleneck experiment data
bottleneck_data = utils.process_Data(bottleneck_data_path)
bottleneck_data = utils.get_2d_velocities(bottleneck_data)

kn = 5

bottleneck_data['mean_spacing'] = bottleneck_data.groupby(['tag', 'frame']).apply(
    lambda group: utils.calc_mean_spacing(group, kn)
).reset_index(level=[0,1], drop=True)
print("Missing values in dataset:\n", bottleneck_data.isna().sum())

bottleneck_data['rel_pos'] = bottleneck_data.groupby(['tag', 'frame']).apply(
    lambda group: utils.calc_relative_positions(group, 5)
).reset_index(level=[0,1], drop=True)
print("Missing values in dataset:\n", bottleneck_data.isna().sum())

print(bottleneck_data.head())

# Processing the corridor experiment data
corridor_data = utils.process_Data(corridor_data_path)
corridor_data = utils.get_2d_velocities(corridor_data)

corridor_data['mean_spacing'] = bottleneck_data.groupby(['tag', 'frame']).apply(
    lambda group: utils.calc_mean_spacing(group, kn)
).reset_index(level=[0,1], drop=True)
print("Missing values in dataset:\n", corridor_data.isna().sum())

corridor_data['rel_pos'] = corridor_data.groupby(['tag', 'frame']).apply(
    lambda group: utils.calc_relative_positions(group, 5)
).reset_index(level=[0,1], drop=True)
print("Missing values in dataset:\n", corridor_data.isna().sum())

print(corridor_data.head())
#TODO: The mean_spacing and rel_pos column contains NaN values for cases that
# did not had enough nbrs to calculate the rel_pos and mean_space, consider this