'''
SCALE - SOURCE AND CLEANING
'''

# library imports
import pandas as pd
import plotly.express as px
import json
import requests
import geopandas as gpd

# cleaned data imports
canada_regions = pd.read_csv('data/canada_regions.csv')
us_regions = pd.read_csv('data/us_regions.csv')
resorts = pd.read_csv('data/resort_cleaned.csv')
weather = pd.read_csv('data/weather_cleaned.csv')

# raw json imports
# json files - countries (https://github.com/datasets/geo-countries/blob/master/data/countries.geojson)
# open raw file
with open('geo/countries.geojson') as file:
    country_json = json.load(file)
    
# retrieve US and Canada from all countries
us_canada_json = []
for country in range(len(country_json['features'])):
    country_name = country_json['features'][country]['properties']['ADMIN']
    if (country_name == 'United States of America') or (country_name == 'Canada'):
        us_canada_json.append(country_json['features'][country].copy())

# rename United States
us_canada_json[1]['properties']['ADMIN'] = 'United States'

# rename ADMIN to name
for country in us_canada_json:
    country['properties']['name'] = country['properties'].pop('ADMIN')

# turn into proper geojson format
us_canada_json = {'features': us_canada_json, 'type': 'FeatureCollection'}
# save
with open('geo/us_canada.json', 'w') as json_file:
    json.dump(us_canada_json, json_file)

# json files - canada provinces (https://github.com/wisdomtheif/Canadian_GeoJSON/blob/master/canada_provinces.geojson?short_path=489f79e)
# open raw file
with open('geo/canada_provinces.geojson') as file:
    canada_provinces_json = json.load(file)

# map for renaming regions
province_rename = {'Newfoundland  & Labrador': 'Newfoundland and Labrador',
                   'Yukon Territory': 'Yukon',
                   }

# loop to rename regions
for province in range(len(canada_provinces_json['features'])):
    province_name = canada_provinces_json['features'][province]['properties']['NAME']
    if province_name in province_rename.keys():
        canada_provinces_json['features'][province]['properties']['NAME'] = province_rename[province_name]

# json files - us states (https://github.com/PublicaMundi/MappingAPI/blob/master/data/geojson/us-states.json?short_path=1c1ebe5)
# open raw file
with open('geo/us_states.json') as file:
    us_states_json = json.load(file)
    
# combine us states and canada provinces/territories
us_canada_state_features = []
for state in range(len(us_states_json['features'])):
    entry = dict()
    entry['geometry'] = us_states_json['features'][state]['geometry']
    entry['properties'] = {'name': us_states_json['features'][state]['properties']['name']}
    entry['type'] = us_states_json['features'][state]['type']
    us_canada_state_features.append(entry)
    
for province in range(len(canada_provinces_json['features'])):
    entry = dict()
    entry['geometry'] = canada_provinces_json['features'][province]['geometry']
    entry['properties'] = {'name': canada_provinces_json['features'][province]['properties']['NAME']}
    entry['type'] = canada_provinces_json['features'][province]['type']
    us_canada_state_features.append(entry)

us_canada_state_province_json = {'features': us_canada_state_features, 'type': 'FeatureCollection'}

with open('geo/us_canada_state_province.json', 'w') as json_file:
    json.dump(us_canada_state_province_json, json_file)
    
# aggregate states and provinces/territories into regions
# turn us_canada_state_province_json into gpd file
gdf = gpd.GeoDataFrame.from_features(us_canada_state_province_json['features'])

# get region and state/province combined dataframe
canada_regions.rename(columns={'province_territory': 'state_province_territory'}, inplace=True)
canada_regions.loc[canada_regions['state_province_territory']=='Yukon Territory', 'state_province_territory'] = 'Yukon'
us_regions.rename(columns={'State Name': 'state_province_territory'}, inplace=True)
us_regions.drop(columns=['Abbreviation'], inplace=True)
us_canada_regions = pd.concat([us_regions, canada_regions], axis=0)

# get list of state/province in each region
region_states_provinces = us_canada_regions.groupby('Region')['state_province_territory'].apply(list).reset_index()

# filter and merge
region_gdf = gpd.GeoDataFrame()
for index, location in region_states_provinces.iterrows():
    region = location['Region']
    region_set = location['state_province_territory']
    # filter locations in new region polygon
    locations_merge = gdf[gdf['name'].isin(region_set)]
    # dissolve polygons into single polygon
    merged_locations = locations_merge.dissolve()
    merged_locations.iloc[0, 1] = region
    # add region_gdf_list
    region_gdf = pd.concat([region_gdf, merged_locations], axis=0)
    
# reset index
region_gdf.reset_index(drop=True, inplace=True)

# save as proper geojson
region_gdf.to_file('geo/regions.geojson', driver='GeoJSON')

'''
Now we have geojson files for three scale levels, each with decreasing areas:
    - country
    - region
    - state_province_territory
'''