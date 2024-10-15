'''
KMEANS CLUSTERING APPLICATIONS
'''

'''
Imports
'''
## LIBRARY IMPORTS ##
# standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import os
import sys
from scipy.spatial.distance import cdist
import importlib.util

# plotly 3d visualization
import plotly.express as px
import plotly.graph_objects as go

# sklearn libraries
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from sklearn.utils import resample

# custom imports
from clustering_kmeans_functions import *

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
SILHOUETTE
'''

# silhouette weather
sil_results_weather = silhouette_kmeans(df_projection=projected_weather,
                                        cluster_min=2,
                                        cluster_max=10,
                                        silhouette_path_start='results/silhouette_results/',
                                        cluster_path_start='results/silhouette_results/',
                                        plot_names='weather',
                                        sample_size=None)

# save sil_samples weather
df_sil_samples_weather = silhouette_to_df(sil_results_weather)
df_sil_samples_weather.to_csv('results/silhouette_results/samples_coefficients_weather.csv', index=False)

# silhouette google
sil_results_google = silhouette_kmeans(df_projection=projected_google,
                                       cluster_min=2,
                                       cluster_max=10,
                                       silhouette_path_start='results/silhouette_results/',
                                       cluster_path_start='results/silhouette_results/',
                                       plot_names='google',
                                       sample_size=None)

# save sil_samples google
df_sil_samples_google = silhouette_to_df(sil_results_google)
df_sil_samples_google.to_csv('results/silhouette_results/samples_coefficients_google.csv', index=False)

# silhouette resorts
sil_results_resorts = silhouette_kmeans(df_projection=projected_resorts,
                                        cluster_min=2,
                                        cluster_max=10,
                                        silhouette_path_start='results/silhouette_results/',
                                        cluster_path_start='results/silhouette_results/',
                                        plot_names='resorts',
                                        sample_size=None)

# save sil_samples resorts
df_sil_samples_resorts = silhouette_to_df(sil_results_resorts)
df_sil_samples_resorts.to_csv('results/silhouette_results/samples_coefficients_resorts.csv', index=False)

# silhouette coefficient plot weather
plot_average_silhouette(sil_results_weather['sil_averages'],
                        cluster_min=2,
                        cluster_max=10,
                        save_path='results/silhouette_results/',
                        plot_name='weather')

# silhouette coefficient plot google
plot_average_silhouette(sil_results_google['sil_averages'],
                        cluster_min=2,
                        cluster_max=10,
                        save_path='results/silhouette_results/',
                        plot_name='google')

# silhouette coefficient plot resorts
plot_average_silhouette(sil_results_resorts['sil_averages'],
                        cluster_min=2,
                        cluster_max=10,
                        save_path='results/silhouette_results/',
                        plot_name='resorts')

'''
ELBOW
'''

# elbow weather
elbow_weather = elbow_kmeans(df_projection=projected_weather,
                             cluster_min=2,
                             cluster_max=10,
                             elbow_path_start='results/elbow_results/',
                             plot_name='weather')

# elbow weather
elbow_google = elbow_kmeans(df_projection=projected_google,
                             cluster_min=2,
                             cluster_max=10,
                             elbow_path_start='results/elbow_results/',
                             plot_name='google')

# elbow weather
elbow_resorts = elbow_kmeans(df_projection=projected_resorts,
                             cluster_min=2,
                             cluster_max=10,
                             elbow_path_start='results/elbow_results/',
                             plot_name='resorts')

'''
VISUALIZATION - RESORTS
'''

# visualize - resorts - 2 clusters
visualize_clustering_results(df_results=results_resorts,
                             clusters=2,
                             label_col='Country',
                             legend_title='Country',
                             main_name='Resorts',
                             save_path='results/kmeans_labels',
                             opacity=0.7)

visualize_clustering_results(df_results=results_resorts,
                             clusters=2,
                             label_col='Region',
                             legend_title='Region',
                             main_name='Resorts',
                             save_path='results/kmeans_labels',
                             opacity=0.7)

visualize_clustering_results(df_results=results_resorts,
                             clusters=2,
                             label_col='Pass',
                             legend_title='Pass',
                             main_name='Resorts',
                             save_path='results/kmeans_labels',
                             opacity=0.7)

# visualize - resorts - 3 clusters
visualize_clustering_results(df_results=results_resorts,
                             clusters=3,
                             label_col='Country',
                             legend_title='Country',
                             main_name='Resorts',
                             save_path='results/kmeans_labels',
                             opacity=0.7)

visualize_clustering_results(df_results=results_resorts,
                             clusters=3,
                             label_col='Region',
                             legend_title='Region',
                             main_name='Resorts',
                             save_path='results/kmeans_labels',
                             opacity=0.7)

visualize_clustering_results(df_results=results_resorts,
                             clusters=3,
                             label_col='Pass',
                             legend_title='Pass',
                             main_name='Resorts',
                             save_path='results/kmeans_labels',
                             opacity=0.7)

# visualize - resorts - 3 clusters
visualize_clustering_results(df_results=results_resorts,
                             clusters=10,
                             label_col='Country',
                             legend_title='Country',
                             main_name='Resorts',
                             save_path='results/kmeans_labels',
                             opacity=0.7)

visualize_clustering_results(df_results=results_resorts,
                             clusters=10,
                             label_col='Region',
                             legend_title='Region',
                             main_name='Resorts',
                             save_path='results/kmeans_labels',
                             opacity=0.7)

visualize_clustering_results(df_results=results_resorts,
                             clusters=10,
                             label_col='Pass',
                             legend_title='Pass',
                             main_name='Resorts',
                             save_path='results/kmeans_labels',
                             opacity=0.7)

'''
VISUALIZATION - WEATHER
'''

# visualize - weather - 2 clusters
visualize_clustering_results(df_results=results_weather,
                             clusters=2,
                             label_col='icon',
                             legend_title='icon',
                             main_name='Weather',
                             save_path='results/kmeans_labels',
                             opacity=0.3,
                             subset=0.005)

# visualize - weather - 3 clusters
visualize_clustering_results(df_results=results_weather,
                             clusters=3,
                             label_col='icon',
                             legend_title='icon',
                             main_name='Weather',
                             save_path='results/kmeans_labels',
                             opacity=0.3,
                             subset=0.005)

# visualize - weather - 5 clusters
visualize_clustering_results(df_results=results_weather,
                             clusters=5,
                             label_col='icon',
                             legend_title='icon',
                             main_name='Weather',
                             save_path='results/kmeans_labels',
                             opacity=0.3,
                             subset=0.005)

# looking at months
results_weather_time_expand = results_weather.copy()
results_weather_time_expand['month'] = results_weather_time_expand['datetime'].dt.month

# visualize - weather - 2 clusters
visualize_clustering_results(df_results=results_weather_time_expand,
                             clusters=2,
                             label_col='month',
                             legend_title='month',
                             main_name='Weather',
                             save_path='results/kmeans_labels',
                             opacity=0.3,
                             subset=0.005)

# visualize - weather - 3 clusters
visualize_clustering_results(df_results=results_weather_time_expand,
                             clusters=3,
                             label_col='month',
                             legend_title='month',
                             main_name='Weather',
                             save_path='results/kmeans_labels',
                             opacity=0.3,
                             subset=0.005)

# visualize - weather - 5 clusters
visualize_clustering_results(df_results=results_weather_time_expand,
                             clusters=5,
                             label_col='month',
                             legend_title='month',
                             main_name='Weather',
                             save_path='results/kmeans_labels',
                             opacity=0.3,
                             subset=0.005)

'''
VISUALIZATION - GOOGLE
'''

# google visualizations - 6 clusters
visualize_clustering_results(df_results=results_google,
                             clusters=4,
                             label_col='Call Category',
                             legend_title='Call Category',
                             main_name='Google',
                             save_path='results/kmeans_labels',
                             opacity=0.3,
                             subset=0.10)

# google visualizations - 6 clusters
visualize_clustering_results(df_results=results_google,
                             clusters=5,
                             label_col='Call Category',
                             legend_title='Call Category',
                             main_name='Google',
                             save_path='results/kmeans_labels',
                             opacity=0.3,
                             subset=0.10)

# google visualizations - 6 clusters
visualize_clustering_results(df_results=results_google,
                             clusters=6,
                             label_col='Call Category',
                             legend_title='Call Category',
                             main_name='Google',
                             save_path='results/kmeans_labels',
                             opacity=0.3,
                             subset=0.10)

# look at merging in country, region, pass
results_google_resorts = results_google.merge(resorts[['Resort', 'Country', 'Region', 'Pass']], on='Resort')

# rerun on new results - 4 clusters
visualize_clustering_results(df_results=results_google_resorts,
                             clusters=4,
                             label_col='Country',
                             legend_title='Country',
                             main_name='Google',
                             save_path='results/kmeans_labels',
                             opacity=0.3,
                             subset=0.10)

visualize_clustering_results(df_results=results_google_resorts,
                             clusters=4,
                             label_col='Region',
                             legend_title='Region',
                             main_name='Google',
                             save_path='results/kmeans_labels',
                             opacity=0.3,
                             subset=0.10)

visualize_clustering_results(df_results=results_google_resorts,
                             clusters=4,
                             label_col='Pass',
                             legend_title='Pass',
                             main_name='Google',
                             save_path='results/kmeans_labels',
                             opacity=0.3,
                             subset=0.10)

# rerun on new results - 5 clusters
visualize_clustering_results(df_results=results_google_resorts,
                             clusters=5,
                             label_col='Country',
                             legend_title='Country',
                             main_name='Google',
                             save_path='results/kmeans_labels',
                             opacity=0.3,
                             subset=0.10)

visualize_clustering_results(df_results=results_google_resorts,
                             clusters=5,
                             label_col='Region',
                             legend_title='Region',
                             main_name='Google',
                             save_path='results/kmeans_labels',
                             opacity=0.3,
                             subset=0.10)

visualize_clustering_results(df_results=results_google_resorts,
                             clusters=5,
                             label_col='Pass',
                             legend_title='Pass',
                             main_name='Google',
                             save_path='results/kmeans_labels',
                             opacity=0.3,
                             subset=0.10)


# rerun on new results - 6 clusters
visualize_clustering_results(df_results=results_google_resorts,
                             clusters=6,
                             label_col='Country',
                             legend_title='Country',
                             main_name='Google',
                             save_path='results/kmeans_labels',
                             opacity=0.3,
                             subset=0.10)

visualize_clustering_results(df_results=results_google_resorts,
                             clusters=6,
                             label_col='Region',
                             legend_title='Region',
                             main_name='Google',
                             save_path='results/kmeans_labels',
                             opacity=0.3,
                             subset=0.10)

visualize_clustering_results(df_results=results_google_resorts,
                             clusters=6,
                             label_col='Pass',
                             legend_title='Pass',
                             main_name='Google',
                             save_path='results/kmeans_labels',
                             opacity=0.3,
                             subset=0.10)
