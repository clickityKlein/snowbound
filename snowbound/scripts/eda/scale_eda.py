'''
SCALE - EDA (PLOTLY CHOROPLETH MAPS)
'''

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import requests
import geopandas as gpd

# cleaned data imports
canada_regions = pd.read_csv('data/canada_regions.csv')
us_regions = pd.read_csv('data/us_regions.csv')
resorts = pd.read_csv('data/resort_cleaned.csv')
weather = pd.read_csv('data/weather_cleaned.csv')

# merge resorts with weather
df = pd.merge(weather, resorts, left_on='resort', right_on='Resort')
df.drop(columns=['Resort'], inplace=True)
df.reset_index(drop=True, inplace=True)
df.rename(columns={'state_province_territory': 'State Province Territory'}, inplace=True)

# deal with datetime properly
df['datetime'] = pd.to_datetime(df['datetime'])
df['day'] = df['datetime'].dt.dayofyear
df['week'] = df['datetime'].dt.isocalendar().week
df['month'] = df['datetime'].dt.month
df['year'] = df['datetime'].dt.year

# cleaned geojson files import
# country scale
with open('geo/us_canada.json') as file:
    country_json = json.load(file)
    
# region scale
with open('geo/regions.geojson') as file:
    region_json = json.load(file)
    
# state/province/territory scale
with open('geo/us_canada_state_province.json') as file:
    state_json = json.load(file)
    
# function to create time scale averages for each spatial scale
def plot_average_choropleth(df, geojson, feature_name, feature_value, save_path, avg_scale='month', id_key='properties.name', duration=1000):
    # gather lat and long
    coordinate_df = df.copy()
    coordinate_df.drop_duplicates(subset=['resort'], inplace=True)
    coordinate_df.reset_index(drop=True, inplace=True)
    
    # group by and average by desired scale
    grouped_df = df.copy()
    grouped_df = grouped_df.groupby([feature_name, avg_scale])[feature_value].mean().reset_index()
    # round feature results to help reduce file size
    grouped_df[feature_value] = grouped_df[feature_value].round(2)
    
    # min and max values for the color scale
    min_value = grouped_df[feature_value].min()
    max_value = grouped_df[feature_value].max()
    
    # choropleth plot
    fig = px.choropleth(grouped_df,
                        locations=feature_name,
                        color=feature_value,
                        geojson=geojson,
                        featureidkey=id_key,
                        scope='north america',
                        projection='mercator',
                        animation_frame=avg_scale,
                        color_continuous_scale='Blues',
                        range_color=(min_value, max_value)
                        )
    
    # resort scatter points
    scatter_points = go.Scattergeo(
        lon=coordinate_df['Longitude'],
        lat=coordinate_df['Latitude'],
        text=coordinate_df['resort'],
        mode='markers',
        marker=dict(color='black', size=4)
    )
    
    # layout options
    fig.update_layout(
        title=f'Average {feature_value.title()} by {feature_name.title()} Across {avg_scale.capitalize()}',
        coloraxis_colorbar=dict(
            title=f'{feature_value.capitalize()}'
        ),
        updatemenus=[{
            'buttons': [{
                'args': [None, {'frame': {'duration': duration, 'redraw': True}, 'fromcurrent': True}],
                'method': 'animate'
            }]
        }]
    )

    fig.add_trace(scatter_points)

    fig.write_html(save_path, include_plotlyjs='cdn', full_html=False)
    
def plot_resort_scatter(df, feature_value, save_path, avg_scale='month', duration=1000, feature_multiplier=1):
    # gather lat and long
    coordinate_df = df.copy()
    coordinate_df.drop_duplicates(subset=['resort'], inplace=True)
    coordinate_df.reset_index(drop=True, inplace=True)
    
    # resort specific groupby and average
    grouped_resort = df.copy()
    grouped_resort = grouped_resort.groupby(['resort', avg_scale])[feature_value].mean().reset_index()
    # round feature results to help reduce file size
    grouped_resort[feature_value] = grouped_resort[feature_value].round(2)
    
    # Create frames for the animation
    frames = []
    for frame_value in grouped_resort[avg_scale].unique():
        frame_df = grouped_resort[grouped_resort[avg_scale] == frame_value]
        scatter_points = go.Scattergeo(
            lon=coordinate_df['Longitude'],
            lat=coordinate_df['Latitude'],
            text=coordinate_df['resort'],
            mode='markers',
            marker=dict(
                size=frame_df[feature_value] * feature_multiplier,
                color='blue',
            )
        )
        frames.append(go.Frame(data=[scatter_points], name=str(frame_value)))
    
    # Initial scatter plot
    initial_frame = grouped_resort[grouped_resort[avg_scale] == grouped_resort[avg_scale].unique()[0]]
    scatter_points = go.Scattergeo(
        lon=coordinate_df['Longitude'],
        lat=coordinate_df['Latitude'],
        text=coordinate_df['resort'],
        mode='markers',
        marker=dict(
            size=initial_frame[feature_value] * feature_multiplier,
            color = 'blue'
        )
    )
    
    fig = go.Figure(data=[scatter_points], frames=frames)
    
    fig.update_layout(
        title=f'Average {feature_value.title()} by Resort Across {avg_scale.capitalize()}',
        geo=dict(
            scope='north america',
            projection=dict(type='mercator')
        ),
        updatemenus=[{
            'buttons': [{
                'args': [None, {'frame': {'duration': duration, 'redraw': True}, 'fromcurrent': True}],
                'method': 'animate',
                'label': 'Play'
            }],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': -0.1,
            'yanchor': 'top'
        }],
        sliders=[{
            'steps': [{
                'args': [[str(frame_value)], {'frame': {'duration': duration, 'redraw': True}, 'mode': 'immediate'}],
                'label': str(frame_value),
                'method': 'animate'
            } for frame_value in grouped_resort[avg_scale].unique()],
            'x': 0.1,
            'len': 0.9,
            'xanchor': 'left',
            'y': -0.2,
            'yanchor': 'top'
        }]
    )

    fig.write_html(save_path, include_plotlyjs='cdn', full_html=False)
    
def create_uniform_outline(geojson_file, save_path):
    # create dataframe matching geojson_file
    feature_names = []
    feature_colors = []
    for feature in geojson_file['features']:
        feature_names.append(feature['properties']['name'])
        feature_colors.append(0)
        
    # create dataframe
    df = pd.DataFrame({'name': feature_names, 'color': feature_colors})
    
    # choropleth plot
    fig = px.choropleth(df,
                        locations='name',
                        geojson=geojson_file,
                        featureidkey='properties.name',
                        scope='north america',
                        projection='mercator',
                        color='color',
                        color_continuous_scale='Blues'
                        )
    
    # hide the scale
    fig.update_layout(coloraxis_showscale=False)
    
    # update hover to just name
    fig.update_traces(hovertemplate='<b>%{location}</b><extra></extra>',
                      text=df['name']
                      )
    
    # save
    fig.write_html(save_path)
    
# run function - snow and months
plot_average_choropleth(df, country_json, 'Country', 'snow', save_path='plots/scale_plots_new/country_month.html', avg_scale='month')
plot_average_choropleth(df, region_json, 'Region', 'snow', save_path='plots/scale_plots_new/region_month.html', avg_scale='month')
plot_average_choropleth(df, state_json, 'State Province Territory', 'snow', save_path='plots/scale_plots_new/state_month.html', avg_scale='month')
plot_resort_scatter(df, 'snow', 'plots/scale_plots_new/resort_month.html', avg_scale='month', duration=1000, feature_multiplier=3)

# run function - snow and weeks
plot_average_choropleth(df, country_json, 'Country', 'snow', save_path='plots/scale_plots_new/country_week.html', avg_scale='week')
plot_average_choropleth(df, region_json, 'Region', 'snow', save_path='plots/scale_plots_new/region_week.html', avg_scale='week')
plot_average_choropleth(df, state_json, 'State Province Territory', 'snow', save_path='plots/scale_plots_new/state_week.html', avg_scale='week')
plot_resort_scatter(df, 'snow', 'plots/scale_plots_new/resort_week.html', avg_scale='week', duration=1000, feature_multiplier=3)

# run function - snow and days - commented out for now, file size becomes too large
# plot_average_choropleth(df, country_json, 'Country', 'snow', save_path='plots/scale_plots/country_day.html', avg_scale='day')
# plot_average_choropleth(df, region_json, 'Region', 'snow', save_path='plots/scale_plots/region_day.html', avg_scale='day')
# plot_average_choropleth(df, state_json, 'State Province Territory', 'snow', save_path='plots/scale_plots/state_day.html', avg_scale='day')
# plot_resort_scatter(df, 'snow', 'plots/scale_plots/resort_day.html', avg_scale='day', duration=1000, feature_multiplier=3)

create_uniform_outline(country_json, 'plots/scale_plots/country_outline.html')
create_uniform_outline(state_json, 'plots/scale_plots/state_outline.html')
create_uniform_outline(region_json, 'plots/scale_plots/region_outline.html')
