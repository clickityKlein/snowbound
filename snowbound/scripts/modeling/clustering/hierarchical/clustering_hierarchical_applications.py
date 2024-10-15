'''
HIERARCHICAL CLUSTERING APPLICATIONS
'''

'''
Imports
'''
# imports
from clustering_hierarchical_functions import *

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
Running Functions
'''
## RESORTS ##
# create linkage matrix
linkage_resorts = create_linkage_matrix(projected_resorts)

# plot dendrogram
'''
results:
    - level 3 with no treshold: 2 clear clusters
    - level 3 with threshold at 30: 3 clear clusters
    - level 4 with threshold at 20: 5 clear clusters
'''
plot_dendrogram(linkage_resorts, level=0, save_path='results/agglomerative/dendrogram_resorts_full.png')
plot_dendrogram(linkage_resorts, level=3, orientation='right', color_threshold=None, save_path='results/agglomerative/dendrogram_resorts_3_0.png')
plot_dendrogram(linkage_resorts, level=3, orientation='right', color_threshold=30, save_path='results/agglomerative/dendrogram_resorts_3_30.png')
plot_dendrogram(linkage_resorts, level=4, orientation='right', color_threshold=20, save_path='results/agglomerative/dendrogram_resorts_4_20.png')

# expand leaves
expanded_resorts_3_0 = expand_leaves(projected_resorts, level=3, color_threshold=None)
expanded_resorts_3_30 = expand_leaves(projected_resorts, level=3, color_threshold=30)
expanded_resorts_4_20 = expand_leaves(projected_resorts, level=4, color_threshold=20)

# analyze leaves - 3_0
analyze_leaves(results_df=results_resorts, parent_nodes=expanded_resorts_3_0, label_col='Country', save_path='results/agglomerative/spread_resorts_3_0_country.png')
analyze_leaves(results_df=results_resorts, parent_nodes=expanded_resorts_3_0, label_col='Region', save_path='results/agglomerative/spread_resorts_3_0_region.png')
analyze_leaves(results_df=results_resorts, parent_nodes=expanded_resorts_3_0, label_col='Pass', save_path='results/agglomerative/spread_resorts_3_0_pass.png')

# analyze leaves - 3_30
analyze_leaves(results_df=results_resorts, parent_nodes=expanded_resorts_3_30, label_col='Country', save_path='results/agglomerative/spread_resorts_3_30_country.png')
analyze_leaves(results_df=results_resorts, parent_nodes=expanded_resorts_3_30, label_col='Region', save_path='results/agglomerative/spread_resorts_3_30_region.png')
analyze_leaves(results_df=results_resorts, parent_nodes=expanded_resorts_3_30, label_col='Pass', save_path='results/agglomerative/spread_resorts_3_30_pass.png')

# analyze leaves - 4_20
analyze_leaves(results_df=results_resorts, parent_nodes=expanded_resorts_4_20, label_col='Country', save_path='results/agglomerative/spread_resorts_4_20_country.png')
analyze_leaves(results_df=results_resorts, parent_nodes=expanded_resorts_4_20, label_col='Region', save_path='results/agglomerative/spread_resorts_4_20_region.png')
analyze_leaves(results_df=results_resorts, parent_nodes=expanded_resorts_4_20, label_col='Pass', save_path='results/agglomerative/spread_resorts_4_20_pass.png')


## WEATHER ##
# create linkage matrix
projected_weather_subset = projected_weather.sample(frac=0.025, random_state=42)
results_weather_subset = results_weather.iloc[projected_weather_subset.index]
projected_weather_subset.reset_index(drop=True, inplace=True)
results_weather_subset.reset_index(drop=True, inplace=True)
linkage_weather = create_linkage_matrix(projected_weather_subset)

# plot dendrogram
'''
results:
    - level 2 with no treshold: 2 clear clusters
    - level 4 with threshold at 155: 3 clear clusters
    - level 4 with threshold at 150: 5 clear clusters
    - level 4 with threshold at 120: 6 clear clusters
'''
plot_dendrogram(linkage_weather, level=2, save_path=None)
plot_dendrogram(linkage_weather, level=4, color_threshold=155, save_path=None)
plot_dendrogram(linkage_weather, level=4, color_threshold=150, save_path=None)
plot_dendrogram(linkage_weather, level=4, color_threshold=120, save_path=None)

# expand leaves
expanded_weather_2_0 = expand_leaves(projected_weather_subset, level=2, color_threshold=None)
expanded_weather_4_155 = expand_leaves(projected_weather_subset, level=4, color_threshold=155)
expanded_weather_4_150 = expand_leaves(projected_weather_subset, level=4, color_threshold=150)
expanded_weather_4_120 = expand_leaves(projected_weather_subset, level=4, color_threshold=120)

# analyze leaves - 2_0
analyze_leaves(results_df=results_weather_subset, parent_nodes=expanded_weather_2_0, label_col='icon', save_path=None)

# analyze leaves - 4_155
analyze_leaves(results_df=results_weather_subset, parent_nodes=expanded_weather_4_155, label_col='icon', save_path=None)

# analyze leaves - 4_150
analyze_leaves(results_df=results_weather_subset, parent_nodes=expanded_weather_4_150, label_col='icon', save_path=None)

# analyze leaves - 4_120
analyze_leaves(results_df=results_weather_subset, parent_nodes=expanded_weather_4_120, label_col='icon', save_path=None)

'''
WEATHER AGGREGATED OVER icon LABEL AND RESORT AND AVERAGED MONTHLY
'''
# aggregation process
weather_monthly = results_weather.copy()
weather_monthly['month'] = weather_monthly['datetime'].dt.month
weather_aggregated = weather_monthly.groupby(['icon', 'resort', 'month'])[['principal_component_1', 'principal_component_2', 'principal_component_3']].mean().reset_index()

# create linkage
linkage_weather_aggregated = create_linkage_matrix(weather_aggregated[['principal_component_1', 'principal_component_2', 'principal_component_3']])

# dendrogram
plot_dendrogram(linkage_weather_aggregated, level=2, save_path=None)
plot_dendrogram(linkage_weather, level=4, color_threshold=155, save_path=None)
plot_dendrogram(linkage_weather, level=4, color_threshold=120, save_path=None)

# expand leaves
expanded_weather_aggregated_2_0 = expand_leaves(weather_aggregated[['principal_component_1', 'principal_component_2', 'principal_component_3']], level=2, color_threshold=None)
expanded_weather_aggregated_4_155 = expand_leaves(weather_aggregated[['principal_component_1', 'principal_component_2', 'principal_component_3']], level=4, color_threshold=155)
expanded_weather_aggregated_4_120 = expand_leaves(weather_aggregated[['principal_component_1', 'principal_component_2', 'principal_component_3']], level=4, color_threshold=120)


# analyze leaves - 2_0
analyze_leaves(results_df=weather_aggregated, parent_nodes=expanded_weather_aggregated_2_0, label_col='icon', save_path=None)

# analyze leaves - 4_155
analyze_leaves(results_df=weather_aggregated, parent_nodes=expanded_weather_aggregated_4_155, label_col='icon', save_path=None)

# analyze leaves - 4_120
analyze_leaves(results_df=weather_aggregated, parent_nodes=expanded_weather_aggregated_4_120, label_col='icon', save_path=None)


## GOOGLE ##
# create linkage matrix
linkage_google = create_linkage_matrix(projected_google)

# plot dendrograms
'''
results:
    - level 3 with threshold at 175: 3 clusters
    - level 4 with threshold at 75: 5 clusters
'''
plot_dendrogram(linkage_google, level=3, color_threshold=175, save_path=None)
plot_dendrogram(linkage_google, level=4, color_threshold=75, save_path=None)
plot_dendrogram(linkage_google, level=7, color_threshold=10, save_path=None)

# expand leaves - 3_175
expanded_google_3_175 = expand_leaves(projected_google, level=3, color_threshold=175)

# expand leaves - 4_75
expanded_google_4_75 = expand_leaves(projected_google, level=4, color_threshold=75)

# analyze leaves - 2_0
analyze_leaves(results_df=results_google, parent_nodes=expanded_google_3_175, label_col='Call Category', save_path=None)

# analyze leaves - 4_75
analyze_leaves(results_df=results_google, parent_nodes=expanded_google_4_75, label_col='Call Category', save_path=None)
