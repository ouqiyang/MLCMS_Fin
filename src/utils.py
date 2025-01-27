import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

def process_Data(data_folder_path):
    data = pd.DataFrame()
    for file in os.listdir(data_folder_path):
        tag = file.split('-')[2].split('.')[0]

        data = pd.read_csv(data_folder_path + '/' + file, header=None, names=['id', 'frame', 'x', 'y', 'z'], sep='\s+')
        data.insert(0, 'tag', tag)

    data.isnull().sum()
    scaler = StandardScaler()
    data[['x', 'y', 'z']] = scaler.fit_transform(data[['x', 'y', 'z']])
    print("Missing values in dataset:\n", data.isnull().sum())
    return data

def get_2d_velocities(data:pd.DataFrame):
    data['velocity_x'] = data.groupby(['id', 'tag'])['x'].diff()*16
    data['velocity_y'] = data.groupby(['id', 'tag'])['y'].diff()*16
    data['speed'] = (data['velocity_x']**2 + data['velocity_y']**2)**0.5
    data.dropna(subset=['velocity_x', 'velocity_y', 'speed'], inplace=True)
    return data

def calc_mean_spacing(group, kn):

    locations = group[["x", "y"]].values

    # Handle cases where the number of points is less than kn + 1
    # (+1 because the nearest neighbor of a point is itself)
    effective_kn = min(kn + 1, len(locations))
    if effective_kn <= 1:
        # If there's only one point in the group, mean_spacing is NaN
        return pd.Series([float('nan')] * len(group), index=group.index)

    nbrs = NearestNeighbors(n_neighbors=effective_kn, algorithm="ball_tree").fit(locations)

    # Compute the distances to the nearest neighbors
    distances, indices = nbrs.kneighbors(locations)

    # Exclude the first column (distance to itself, which is zero)
    mean_distances = distances[:, 1:].mean(axis=1)

    return pd.Series(mean_distances, index=group.index)

def calc_relative_positions(group, kn):

    locations = group[["x", "y"]].values

    num_points = len(locations)
    # Handle cases where the number of points is less than kn + 1
    effective_kn = min(kn + 1, num_points)
    if effective_kn <= 1:
        # If there's only one point in the group, relative positions are NaN
        return pd.Series([float('nan')] * num_points, index=group.index)

    nbrs = NearestNeighbors(n_neighbors=effective_kn, algorithm="ball_tree").fit(locations)
    
    _, indices = nbrs.kneighbors(locations)
    
    # Compute relative positions
    rels = locations[indices[:, 1:]] - locations[:, np.newaxis, :]
    rel_positions = rels.tolist()
    rel_positions = [ [tuple(rel) for rel in rel_point] for rel_point in rel_positions ]
    
    return pd.Series(rel_positions, index=group.index)
