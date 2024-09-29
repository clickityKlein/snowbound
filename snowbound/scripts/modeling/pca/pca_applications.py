'''
PCA - APPLICATIONS
'''

'''
Imports
'''
## LIBRARY IMPORTS ##
# standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# plotly 3d visualization
import plotly.express as px
import plotly.graph_objects as go

# sklearn libraries
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# custom pca functions
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

'''
General PCA
'''
## GENERAL PCA - WEATHER ##
# run pca
projection_weather_gen, explained_weather_gen, components_weather_gen, loadings_weather_gen, results_weather_gen, eigen_weather_gen = perform_pca(weather, quant_cols_weather, label_cols_weather, n_components=None)

# validate orthogonality
validate = validate_orthogonality(components_weather_gen)

# visualize variance
visualize_variance(explained_weather_gen, title_suffix='General Weather', save_path='results/explained_variance_weather_gen.png')

# visualize variance - 95% retention
visualize_variance(explained_weather_gen, title_suffix='General Weather', threshold=0.95, save_path='results/explained_variance_weather_gen_95.png')

# save variance dataframe
explained_weather_gen.to_csv('results/explained_weather_gen.csv')

# save loadings matrix
loadings_weather_gen.to_csv('results/loadings_weather_gen.csv')

# save eigenvalues
eigen_weather_gen.to_csv('results/eigen_weather_gen.csv')

# visualize loadings matrix rank
rank_and_visualize_loadings(loadings_weather_gen, plot_type='barplot', save_path='results/loadings_matrix_rank_barplot_weather_gen.png')
rank_and_visualize_loadings(loadings_weather_gen, plot_type='boxplot', save_path='results/loadings_matrix_rank_boxplot_weather_gen.png')

## GENERAL PCA - Google ##
# run pca
projection_google_gen, explained_google_gen, components_google_gen, loadings_google_gen, results_google_gen, eigen_google_gen = perform_pca(google, quant_cols_google, label_cols_google, n_components=None)

# validate orthogonality
validate = validate_orthogonality(components_weather_gen)

# visualize variance
visualize_variance(explained_google_gen, title_suffix='General Google', save_path='results/explained_variance_google_gen.png')

# visualize variance - 95% retention
visualize_variance(explained_google_gen, title_suffix='General Google', threshold=0.95, save_path='results/explained_variance_google_gen_95.png')

# save variance dataframe
explained_google_gen.to_csv('results/explained_google_gen.csv')

# save loadings matrix
loadings_google_gen.to_csv('results/loadings_google_gen.csv')

# save eigenvalues
eigen_google_gen.to_csv('results/eigen_google_gen.csv')

# visualize loadings matrix rank
rank_and_visualize_loadings(loadings_google_gen, plot_type='barplot', save_path='results/loadings_matrix_rank_barplot_google_gen.png')
rank_and_visualize_loadings(loadings_google_gen, plot_type='boxplot', save_path='results/loadings_matrix_rank_boxplot_google_gen.png')

## GENERAL PCA - Resorts ##
# run pca
projection_resorts_gen, explained_resorts_gen, components_resorts_gen, loadings_resorts_gen, results_resorts_gen, eigen_resorts_gen = perform_pca(resorts, quant_cols_resorts, label_cols_resorts, n_components=None)

# validate orthogonality
validate = validate_orthogonality(components_resorts_gen)

# visualize variance
visualize_variance(explained_resorts_gen, title_suffix='General Resorts', save_path='results/explained_variance_resorts_gen.png')

# visualize variance - 95% retention
visualize_variance(explained_resorts_gen, title_suffix='General Resorts', threshold=0.95, save_path='results/explained_variance_resorts_gen_95.png')

# save variance dataframe
explained_resorts_gen.to_csv('results/explained_resorts_gen.csv')

# save loadings matrix
loadings_resorts_gen.to_csv('results/loadings_resorts_gen.csv')

# save eigenvalues
eigen_resorts_gen.to_csv('results/eigen_resorts_gen.csv')

# visualize loadings matrix rank
rank_and_visualize_loadings(loadings_resorts_gen, plot_type='barplot', save_path='results/loadings_matrix_rank_barplot_resorts_gen.png')
rank_and_visualize_loadings(loadings_resorts_gen, plot_type='boxplot', save_path='results/loadings_matrix_rank_boxplot_resorts_gen.png')


'''
3-Dimensional PCA
'''
## 3D PCA - WEATHER ##
# run pca
projection_weather_3d, explained_weather_3d, components_weather_3d, loadings_weather_3d, results_weather_3d, eigen_weather_3d = perform_pca(weather, quant_cols_weather, label_cols_weather, n_components=3)

# visualize variance
visualize_variance(explained_weather_3d, title_suffix='3D Weather', save_path='results/explained_variance_weather_3d.png')

# save variance dataframe
explained_weather_3d.to_csv('results/explained_weather_3d.csv')

# save loadings matrix
loadings_weather_3d.to_csv('results/loadings_weather_3d.csv')

# visualize loadings matrix rank
rank_and_visualize_loadings(loadings_weather_3d, plot_type='barplot', save_path='results/loadings_matrix_rank_barplot_weather_3d.png')
rank_and_visualize_loadings(loadings_weather_3d, plot_type='boxplot', save_path='results/loadings_matrix_rank_boxplot_weather_3d.png')

# static 3d visualization
# set up for weather - use December 2023
df_results = results_weather_3d.copy()
df_results_subset = df_results[(df_results['datetime'].dt.year==2023) & (df_results['datetime'].dt.month==12)]

# run static visualization function
visualize_results_3d(df_results_subset, label_col='icon', legend_title='Weather Type', save_path='results/weather_3d_vis.html', arrow_size=15)

# animated 3d visualization
# set up - no changes
df_results = results_weather_3d.copy()

# run animated visualization function
animate_results_timeseries_3d(df_results, label_col='icon', legend_title='Weather Type', save_path='results/weather_3d_animation.html', arrow_size=15)

## 3D PCA - GOOGLE ##
# run pca
projection_google_3d, explained_google_3d, components_google_3d, loadings_google_3d, results_google_3d, eigen_google_3d = perform_pca(google, quant_cols_google, label_cols_google, n_components=3)

# visualize variance
visualize_variance(explained_google_3d, title_suffix='3D Google', save_path='results/explained_variance_google_3d.png')

# save variance dataframe
explained_google_3d.to_csv('results/explained_google_3d.csv')

# save loadings matrix
loadings_google_3d.to_csv('results/loadings_google_3d.csv')

# visualize loadings matrix rank
rank_and_visualize_loadings(loadings_google_3d, plot_type='barplot', save_path='results/loadings_matrix_rank_barplot_google_3d.png')
rank_and_visualize_loadings(loadings_google_3d, plot_type='boxplot', save_path='results/loadings_matrix_rank_boxplot_google_3d.png')

# run static visualization function
visualize_results_3d(results_google_3d, label_col='Call Category', legend_title='Call Category', save_path='results/google_3d_vis.html', arrow_size=30)

## 3D PCA - RESORTS ##
# run pca
projection_resorts_3d, explained_resorts_3d, components_resorts_3d, loadings_resorts_3d, results_resorts_3d, eigen_resorts_3d = perform_pca(resorts, quant_cols_resorts, label_cols_resorts, n_components=3)

# visualize variance
visualize_variance(explained_resorts_3d, title_suffix='3D Resorts', save_path='results/explained_variance_resorts_3d.png')

# save variance dataframe
explained_resorts_3d.to_csv('results/explained_resorts_3d.csv')

# save loadings matrix
loadings_resorts_3d.to_csv('results/loadings_resorts_3d.csv')

# visualize loadings matrix rank
rank_and_visualize_loadings(loadings_resorts_3d, plot_type='barplot', save_path='results/loadings_matrix_rank_barplot_resorts_3d.png')
rank_and_visualize_loadings(loadings_resorts_3d, plot_type='boxplot', save_path='results/loadings_matrix_rank_boxplot_resorts_3d.png')

# run static visualization function
# region label
visualize_results_3d(results_resorts_3d, label_col='Region', legend_title='Region', save_path='results/resorts_region_3d_vis.html')
# pass label
visualize_results_3d(results_resorts_3d, label_col='Pass', legend_title='Pass', save_path='results/resorts_pass_3d_vis.html')
# country
visualize_results_3d(results_resorts_3d, label_col='Country', legend_title='Country', save_path='results/resorts_country_3d_vis.html')


'''
2-Dimensional PCA
'''
## 2D PCA - WEATHER ##
# run pca
projection_weather_2d, explained_weather_2d, components_weather_2d, loadings_weather_2d, results_weather_2d, eigen_weather_2d = perform_pca(weather, quant_cols_weather, label_cols_weather, n_components=2)

# visualize variance
visualize_variance(explained_weather_2d, title_suffix='2D Weather', save_path='results/explained_variance_weather_2d.png')

# save variance dataframe
explained_weather_2d.to_csv('results/explained_weather_2d.csv')

# save loadings matrix
loadings_weather_2d.to_csv('results/loadings_weather_2d.csv')

# visualize loadings matrix rank
rank_and_visualize_loadings(loadings_weather_2d, plot_type='barplot', save_path='results/loadings_matrix_rank_barplot_weather_2d.png')
rank_and_visualize_loadings(loadings_weather_2d, plot_type='boxplot', save_path='results/loadings_matrix_rank_boxplot_weather_2d.png')

# run static visualization function
visualize_results_2d(results_weather_2d, label_col='icon', legend_title='Weather Type', title_suffix='Weather', save_path='results/weather_2d_vis.png')

## 2D PCA - GOOGLE ##
# run pca
projection_google_2d, explained_google_2d, components_google_2d, loadings_google_2d, results_google_2d, eigen_google_2d = perform_pca(google, quant_cols_google, label_cols_google, n_components=2)

# visualize variance
visualize_variance(explained_google_2d, title_suffix='2D Google', save_path='results/explained_variance_google_2d.png')

# save variance dataframe
explained_google_2d.to_csv('results/explained_google_2d.csv')

# save loadings matrix
loadings_google_2d.to_csv('results/loadings_google_2d.csv')

# visualize loadings matrix rank
rank_and_visualize_loadings(loadings_google_2d, plot_type='barplot', save_path='results/loadings_matrix_rank_barplot_google_2d.png')
rank_and_visualize_loadings(loadings_google_2d, plot_type='boxplot', save_path='results/loadings_matrix_rank_boxplot_google_2d.png')

# run static visualization function
visualize_results_2d(results_google_2d, label_col='Call Category', legend_title='Call Category', title_suffix='Google', save_path='results/google_2d_vis.png')

## 2D PCA - Resorts ##
# run pca
projection_resorts_2d, explained_resorts_2d, components_resorts_2d, loadings_resorts_2d, results_resorts_2d, eigen_resorts_2d = perform_pca(resorts, quant_cols_resorts, label_cols_resorts, n_components=2)

# visualize variance
visualize_variance(explained_resorts_2d, title_suffix='2D Resorts', save_path='results/explained_variance_resorts_2d.png')

# save variance dataframe
explained_resorts_2d.to_csv('results/explained_resorts_2d.csv')

# save loadings matrix
loadings_resorts_2d.to_csv('results/loadings_resorts_2d.csv')

# visualize loadings matrix rank
rank_and_visualize_loadings(loadings_resorts_2d, plot_type='barplot', save_path='results/loadings_matrix_rank_barplot_resorts_2d.png')
rank_and_visualize_loadings(loadings_resorts_2d, plot_type='boxplot', save_path='results/loadings_matrix_rank_boxplot_resorts_2d.png')

# run static visualization function
# region
visualize_results_2d(results_resorts_2d, label_col='Region', legend_title='Region', title_suffix='Resorts by Region', save_path='results/resorts_region_2d_vis.png')
# pass
visualize_results_2d(results_resorts_2d, label_col='Pass', legend_title='Pass', title_suffix='Resorts by Pass', save_path='results/resorts_pass_2d_vis.png')
# country
visualize_results_2d(results_resorts_2d, label_col='Country', legend_title='Country', title_suffix='Resorts by Country', save_path='results/resorts_country_2d_vis.png')
