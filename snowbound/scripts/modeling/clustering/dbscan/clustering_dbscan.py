'''
DBSCAN TESTING
'''

# standard libraries
import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

# clustering libraries
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

# epsilon search
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

# plotly 3d visualization
import plotly.express as px
import plotly.graph_objects as go

# custom pca functions
sys.path.insert(0, '../pca')
from pca_functions import *

## DATA IMPORTS ##
# relative path to data
relative_path = '../../data'

# weather
weather_path = os.path.join(relative_path, 'weather_cleaned.csv')
weather = pd.read_csv(weather_path)
weather['datetime'] = pd.to_datetime(weather['datetime'])

# google places
google_path = os.path.join(relative_path, 'google_cleaned.csv')
google = pd.read_csv(google_path)

# resorts
resort_path = os.path.join(relative_path, 'resort_cleaned.csv')
resorts = pd.read_csv(resort_path)

## QUANTITATIVE AND LABEL COLUMN(S) ##
# weather
quant_cols_weather = ['tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike', 'dew', 'humidity', 'precip', 'snow', 'snowdepth', 'windgust', 'windspeed', 'winddir', 'pressure', 'cloudcover', 'visibility', 'solarradiation', 'solarenergy', 'uvindex', 'moonphase', 'severerisk']
label_cols_weather = ['datetime', 'icon', 'resort', 'type_snow', 'type_rain', 'type_ice', 'type_freezingrain', 'type_none']

# google places
quant_cols_google = ['Latitude', 'Longitude', 'rating', 'total_ratings']
label_cols_google = ['Name', 'Resort', 'Call Category', 'Initial Category', 'Secondary Category', 'Tertiary Category']

# resorts
quant_cols_resorts = ['Overall Rating', 'Elevation Difference', 'Elevation Low', 'Elevation High', 'Trails Total', 'Trails Easy', 'Trails Intermediate', 'Trails Difficult', 'Lifts', 'Price', 'Resort Size', 'Run Variety', 'Lifts Quality', 'Latitude', 'Longitude']
label_cols_resorts = ['Resort', 'state_province_territory', 'Country', 'City', 'Pass', 'Region']

## PERFORM PCA ##

# run custom pca
pca_weather = perform_pca(weather, quant_cols_weather, label_cols_weather, n_components=3)
pca_google = perform_pca(google, quant_cols_google, label_cols_google, n_components=3)
pca_resorts = perform_pca(resorts, quant_cols_resorts, label_cols_resorts, n_components=3)

# extract projected components
projected_weather = pca_weather[0]
results_weather = pca_weather[4]
projected_google = pca_google[0]
results_google = pca_google[4]
projected_resorts = pca_resorts[0]
results_resorts = pca_resorts[4]

'''
DBSCAN PLAYGROUND

epsilon and mininum sample heuristic (outside domain knowledge):
    - epsilon: k-distance graph with elbow points
    - minimum sample: number of dimensions + 1
'''

# function to find epsilon based on a common heuristic
def choose_epsilon(projection_df, name, neighbors=3, save_path=None):
    '''
    An automatic detection of elbow point using kNN.

    Parameters
    ----------
    projection_df : pandas DataFrame
        Data projected onto a PCA space (created with 3D PCA concepts).
    name : string
        Name of dataset, for illustrative and saving purposes.
    neighbors : int, optional
        Number of neighbors to run with epsilon. The default is 3.
    save_path : string, optional
        If save path, save in this directory. The default is None.

    Returns
    -------
    elbow_distance : float
        Recommended epsilon based on PCA projection and specified neighbors.

    '''
    
    # run k-NN
    nbrs = NearestNeighbors(n_neighbors=neighbors).fit(projection_df)
    distances, indices = nbrs.kneighbors(projection_df)
    distances = np.sort(distances[:, neighbors-1], axis=0)
    
    # find elbow
    kneedle = KneeLocator(range(len(distances)), distances, curve='convex', direction='increasing')
    elbow_point = kneedle.elbow
    elbow_distance = distances[elbow_point]
    
    # plot
    plt.plot(distances, label='Distance')
    plt.axhline(y=distances[elbow_point], color='r', linestyle='--', label=f'Elbow: ({elbow_point}, {round(distances[elbow_point], 2)})')
    plt.xlabel('Points')
    plt.ylabel('k-distance')
    plt.title(f'k-distance Graph: {name.title()}')
    plt.legend()
    
    if save_path:
        plt.savefig(f'{save_path}/auto_eps_{name}.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # return elbow
    return elbow_distance

def choose_epsilon_manual(projection_df, name, neighbors=3, save_path=None):
    '''
    A more manual detection of elbow point using clustering on the slopes of the kNN plot.

    Parameters
    ----------
    projection_df : pandas DataFrame
        Data projected onto a PCA space (created with 3D PCA concepts).
    name : string
        Name of dataset, for illustrative and saving purposes.
    neighbors : int, optional
        Number of neighbors to run with epsilon. The default is 3.
    save_path : string, optional
        If save path, save in this directory. The default is None.

    Returns
    -------
    slopes_df : pandas DataFrame
        Slopes of the kNN plot.
    distances : numpy Array
        Distances between slopes.

    '''
    
    # run k-NN with distances
    nbrs = NearestNeighbors(n_neighbors=neighbors).fit(projection_df)
    distances, indices = nbrs.kneighbors(projection_df)
    distances = np.sort(distances[:, neighbors-1], axis=0)
    
    # find slope between distances
    slopes = []
    for distance in range(len(distances)):
        if distance == 0:
            slopes.append(0)
        else:
            slopes.append(distances[distance] - distances[distance -1])
    
    # run kmeans on slope - set to 3
    model = KMeans(n_clusters=3)
    model.fit(np.array(slopes).reshape(-1, 1))
    labels = model.labels_
    slopes_df = pd.DataFrame({'slope': slopes, 'label': labels})
    slopes_df.reset_index(inplace=True)
    slopes_df.rename(columns={'index': 'slope_index'}, inplace=True)
    
    # plotting
    sns.scatterplot(slopes_df, x='slope_index', y='slope', hue='label')
    plt.title(f'k-distance Slope Graph: {name.title()}')
    plt.xlabel('Points')
    plt.ylabel('k-distance Slope')
    
    if save_path:
        plt.savefig(f'{save_path}/manual_eps_{name}.png', dpi=300, bbox_inches='tight')
    
    plt.show()
            
    return slopes_df, distances

def visualize_dbscan_results(projection_df, clusters, main_name, save_path, opacity=0.7):
    '''
    Creates an html object of DBSCAN labels onto a PCA space.
    
    Parameters
    ----------
    projection_df : pandas DataFrame
        Data projected onto a PCA space (created with 3D PCA concepts).
    clusters : numpy Array
        Labels from a DBSCAN object.
    main_name : string
        Name of dataset, for illustrative and saving purposes..
    save_path : string, optional
        If save path, save in this directory. The default is None..
    opacity : float, optional
        Set opacity for the visual result. The default is 0.7.

    Returns
    -------
    None.

    '''
    
    # create a copy of df_results
    df = projection_df.copy()
    
    # mapping cluster labels
    mapping = {-1: 'Outlier'}
    mapping.update({val: f'Cluster {cluster_i+1}' for cluster_i, val in enumerate(set(clusters)) if val != -1})
    mapped_values = [mapping[val] for val in clusters]
    mapped_df = pd.DataFrame({'cluster': mapped_values})
    
    # add column to df
    df = pd.concat([df, mapped_df], axis=1)
    df['cluster'] = pd.Categorical(df['cluster'], categories=sorted(mapping.values(), key=lambda x: (x != 'Outlier', x)))
    df.sort_values(by='cluster', inplace=True)
    
    # color sequence
    color_sequence = ['black'] + px.colors.qualitative.Plotly[1:]
    
    # begin figure
    fig = px.scatter_3d(df, x='principal_component_1', y='principal_component_2', z='principal_component_3', color='cluster', opacity=opacity, color_discrete_sequence=color_sequence)
    
    # disable layout panes
    fig.update_layout(scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False)
    ))
    
    # update legend
    fig.update_layout(legend=dict(
        title='Clusters',
        x=0.1,  # Position of the legend
        y=0.9,
        bgcolor='rgba(255, 255, 255, 0.5)',  # Background color with transparency
        bordercolor='black',
        borderwidth=2
    ))
    
    # write plotly figure to html
    fig.write_html(f'{save_path}/{main_name.lower()}.html')
    

## RESORTS ##
# find epsilon - automatic
epsilon_auto_resorts = choose_epsilon(projected_resorts, 'Resorts', save_path='results/dbscan')

# find epsilon - manual
slopes_resorts, distances_resorts = choose_epsilon_manual(projected_resorts, 'Resorts', save_path='results/dbscan')
epsilon_manual_resorts = distances_resorts[slopes_resorts[slopes_resorts['label']==2].idxmin()[0]]

# run dbscan - automatic
dbscan_resorts_auto = DBSCAN(eps=epsilon_auto_resorts, min_samples=3).fit(projected_resorts)
labels_resorts_auto = dbscan_resorts_auto.labels_
visualize_dbscan_results(projected_resorts, labels_resorts_auto, 'Resorts_auto', 'results/dbscan', opacity=0.7)

# run dbscan - manual
dbscan_resorts_manual = DBSCAN(eps=epsilon_manual_resorts, min_samples=3).fit(projected_resorts)
labels_resorts_manual = dbscan_resorts_manual.labels_
visualize_dbscan_results(projected_resorts, labels_resorts_manual, 'Resorts_manual', 'results/dbscan', opacity=0.7)


## WEATHER ##
# subset weather
projected_weather_subset = projected_weather.sample(frac=0.05, random_state=42)
projected_weather_subset.reset_index(drop=True, inplace=True)
# find epsilon - automatic
epsilon_auto_weather = choose_epsilon(projected_weather_subset, 'Weather', save_path='results/dbscan')

# find epsilon - manual
slopes_weather, distances_weather = choose_epsilon_manual(projected_weather_subset, 'Weather', save_path='results/dbscan')
epsilon_manual_weather = distances_weather[slopes_weather[slopes_weather['label']==2].idxmin()[0]]

# run dbscan - automatic
dbscan_weather_auto = DBSCAN(eps=epsilon_auto_weather, min_samples=3).fit(projected_weather_subset)
labels_weather_auto = dbscan_weather_auto.labels_
visualize_dbscan_results(projected_weather_subset, labels_weather_auto, 'weather_auto', 'results/dbscan', opacity=0.3)

# run dbscan - manual
dbscan_weather_manual = DBSCAN(eps=epsilon_manual_weather, min_samples=3).fit(projected_weather_subset)
labels_weather_manual = dbscan_weather_manual.labels_
visualize_dbscan_results(projected_weather_subset, labels_weather_manual, 'weather_manual', 'results/dbscan', opacity=0.3)


## GOOGLE ##
# find epsilon - automatic
epsilon_auto_google = choose_epsilon(projected_google, 'Google', save_path='results/dbscan')

# find epsilon - manual
slopes_google, distances_google = choose_epsilon_manual(projected_google, 'Google', save_path='results/dbscan')
epsilon_manual_google = distances_google[slopes_google[slopes_google['label']==2].idxmin()[0]]

# run dbscan - automatic
dbscan_google_auto = DBSCAN(eps=epsilon_auto_google, min_samples=3).fit(projected_google)
labels_google_auto = dbscan_google_auto.labels_
visualize_dbscan_results(projected_google, labels_google_auto, 'google_auto', 'results/dbscan', opacity=0.7)

# run dbscan - manual
dbscan_google_manual = DBSCAN(eps=epsilon_manual_google, min_samples=3).fit(projected_google)
labels_google_manual = dbscan_google_manual.labels_
visualize_dbscan_results(projected_google, labels_google_manual, 'google_manual', 'results/dbscan', opacity=0.7)

# run dbscan - custom
dbscan_google_custom = DBSCAN(eps=0.65, min_samples=10).fit(projected_google)
labels_google_custom = dbscan_google_custom.labels_
visualize_dbscan_results(projected_google, labels_google_custom, 'google_custom', 'results/dbscan', opacity=0.7)
