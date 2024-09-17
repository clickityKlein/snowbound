# eda/eda_folium

'''
Module for mapping with the folium library.
'''

import pandas as pd
import folium
import os
import ast

'''
Functions created in cleaning_ski_resorts
'''
# function to import data which accounts for kaggle dataset
def import_data(relative_path, file):
    # create retrieve file path
    file_path = os.path.join(relative_path, f'{file}.csv')
    
    # try to open file
    try:
        df = pd.read_csv(file_path)
    except:
        try:
            df = pd.read_csv(file_path, encoding='latin1')
        except:
            print('File Does Not Exist')
            return None
    
    return df


'''
Return information about weather and stations on a given day.
'''
# function to unpack weather stations for a single day of data
def unpack_stations(weather_df, station_df):
    weather_df = weather_df.copy()
    station_df = station_df.copy()
    
    resort_stations = {'Resort': [],
                       'Station': [],
                       'Latitude': [],
                       'Longitude': [],
                       'Distance': []}
    
    for col, row in weather_df.iterrows():
        resort = row['resort']
        stations = row['stations']
        
        if type(stations) != list:
            stations = ast.literal_eval(stations)
            
        for station in stations:
            # get last station object
            selection = station_df[(station_df['id']==station) & (station_df['resort']==resort)]
            
            if selection.size != 0:
                selection = selection.iloc[-1]
            
                # get information
                lat = selection['latitude']
                long = selection['longitude']
                distance = selection['distance']
                
                # append results
                resort_stations['Resort'].append(resort)
                resort_stations['Station'].append(station)
                resort_stations['Latitude'].append(lat)
                resort_stations['Longitude'].append(long)
                resort_stations['Distance'].append(distance)
    
    df = pd.DataFrame(resort_stations)
    
    return df


'''
folium complex mapping with dataframe
'''
def folium_map_complex(map_df, map_str, min_width=300, max_width=300, supporting_radius=5):
    # create map - center on "main" df
    main_df = map_df[map_df['main']]['df'].values[0]
    
    # use main_df information to create map object
    f_map = folium.Map(location=[main_df['Latitude'].mean(), main_df['Longitude'].mean()], zoom_start=3)
    
    # creating toggling groups
    groups = dict()
    for col, row in map_df.iterrows():
        group = row['group']
        groups[f'{group}_group'] = folium.FeatureGroup(name=group).add_to(f_map)
        
    # add radius group manually
    radius_group = folium.FeatureGroup(name='Station Radius').add_to(f_map)
    
    # add locations
    for col, row in map_df.iterrows():
        if row['main']:
            # get main attributes
            df = row['df']
            group = row['group']
            color = row['color']
            radius_color = row['radius_color']
            for col, row in df.iterrows():
                # main popup
                iframe = folium.IFrame(row['popup'])
                
                # initialize popup
                popup = folium.Popup(iframe, min_width=min_width, max_width=max_width)
                
                # add each row
                folium.Marker(location=[row['Latitude'], row['Longitude']], popup=popup).add_to(groups[f'{group}_group'])
                
                # add radius
                folium.Circle(
                    location = [row['Latitude'], row['Longitude']],
                    radius = row['Distance'],
                    color = color,
                    fill_color = radius_color,
                    fill = True,
                    fill_opacity = 0.3
                    ).add_to(radius_group)
        else:
            # get supporting attributes
            df = row['df']
            group = row['group']
            color = row['color']
            
            for col, row in df.iterrows():
                # popup settings
                # iframe = folium.IFrame(row['popup'].replace('\n', '<br><br>'))
                iframe = folium.IFrame(row['popup'])
                popup = folium.Popup(iframe, min_width=min_width, max_width=max_width)
                
                folium.CircleMarker(
                    location = [row['Latitude'], row['Longitude']],
                    radius = supporting_radius,
                    color = color,
                    fill_color = color,
                    fill = True,
                    fill_opacity = 1,
                    popup = popup
                    ).add_to(groups[f'{group}_group'])
            
    # add layer toggle control
    folium.LayerControl().add_to(f_map)
    
    # save map
    return f_map.save(f'{map_str}.html')


'''
import data and create map
'''

## IMPORT DATA ## 
relative_path = '../../data'

# resorts
mapping_resorts = import_data(relative_path, 'resort_cleaned')

# google places
mapping_google = import_data(relative_path, 'google_cleaned')

# weather
mapping_weather = import_data(relative_path, 'weather_cleaned')
# make sure datetime has proper datetime format
mapping_weather['datetime'] = pd.to_datetime(mapping_weather['datetime'])

# stations
mapping_stations = import_data(relative_path, 'stations_cleaned')


## RUN UNPACK STATIONS ON MAX DATE ##
# max date
max_date = mapping_weather['datetime'].max()

# max date subset
weather_max = mapping_weather[mapping_weather['datetime']==max_date]

# run the function
resort_stations = unpack_stations(weather_max, mapping_stations)


## MAPPING SETUP ##

# resort popup
def resort_popup(resort, city, state_province, pass_type):
    popup_string = f'<h1>{resort}</h1><br>Location: {city}, {state_province}<br>Pass: {pass_type}'
    
    return popup_string

mapping_resorts['popup'] = mapping_resorts.apply(lambda row: resort_popup(row['Resort'],
                                                                          row['City'],
                                                                          row['state_province_territory'],
                                                                          row['Pass']),
                                                 axis=1)

# reset index
mapping_resorts.reset_index(drop=True, inplace=True)

# station popup
resort_stations['popup'] = '<h1>Weather Station</h1>'

# merge distance into stations
resort_station_max = resort_stations.groupby(['Resort'])['Distance'].max().reset_index()
resort_complete_df = pd.merge(mapping_resorts, resort_station_max, on='Resort')

# remove stations associated with any missing resorts
map_station_df = resort_stations[resort_stations['Resort'].isin(resort_complete_df['Resort'].unique())]

# remove duplicate coordinates and reset index
map_station_df.drop_duplicates(subset=['Latitude', 'Longitude'], inplace=True)
map_station_df.reset_index(drop=True, inplace=True)

# get google call categories
'''
'Restaurants'
'Bars'
'Spas'
'Shopping'
'Medical'
'Grocery'
'Lodging'
'''

def create_google_df_popup(name, call, initial, secondary, tertiary):
    # create dictionary
    popup_dict = {'Resort': name.title(),
                  'Category Call': call,
                  'Initial Return Category': initial.replace('_', ' ').title(),
                  'Secondary Return Category': secondary.replace('_', ' ').title(),
                  'Tertiary Return Category': tertiary.replace('_', ' ').title()}
    
    # create dataframe
    popup_df = pd.DataFrame(list(popup_dict.items()), columns = ['Category', 'Result']).set_index('Category')
    
    # turn dataframe to html style
    popup_html = popup_df.to_html(
        classes = "table table-striped table-hover table-condensed table-responsive"
        )
    
    return popup_html

def create_google_table(name, call, initial, secondary, tertiary, rating, total_ratings):
    # create dictionary
    popup_html = '<table border="1" class="dataframe table table-striped table-hover table-condensed table-responsive">'
    popup_html += '<thead><tr><th>Category</th><th>Result</th></thead>'
    popup_html += '<tbody>'
    popup_html += f'<tr><td>Business</td><td>{name}</td></tr>'
    popup_html += f'<tr><td>Category Call</td><td>{call}</td></tr>'
    popup_html += f'<tr><td>Initial Return</td><td>{initial.replace("_", " ").title()}</td></tr>'
    popup_html += f'<tr><td>Secondary Return</td><td>{secondary.replace("_", " ").title()}</td></tr>'
    popup_html += f'<tr><td>Tertiary Return</td><td>{tertiary.replace("_", " ").title()}</td></tr>'
    popup_html += f'<tr><td>Rating</td><td>{rating}</td></tr>'
    popup_html += f'<tr><td>Total Ratings</td><td>{int(total_ratings)}</td></tr>'
    popup_html += '</tbody></table>'
    
    return popup_html

        
mapping_google['popup'] = mapping_google.apply(lambda row: create_google_table(row['Name'],
                                                                               row['Call Category'],
                                                                               row['Initial Category'],
                                                                               row['Secondary Category'],
                                                                               row['Tertiary Category'],
                                                                               row['rating'],
                                                                               row['total_ratings']),
                                               axis=1)


'''
'Restaurants'
'''
# subset google on call categories and take top 5
restaurants_df = mapping_google[mapping_google['Call Category']=='Restaurants']

# get top 5 by rating for each resort
restaurants_df = restaurants_df.groupby(['Resort']).apply(lambda row: row.nlargest(5, 'rating')).reset_index(drop=True)


'''
'Bars'
'''
# subset google on call categories and take top 5
bars_df = mapping_google[mapping_google['Call Category']=='Bars']

# get top 5 by rating for each resort
bars_df = bars_df.groupby(['Resort']).apply(lambda row: row.nlargest(5, 'rating')).reset_index(drop=True)

'''
'Spas'
'''
# subset google on call categories and take top 5
spas_df = mapping_google[mapping_google['Call Category']=='Spas']

# get top 5 by rating for each resort
spas_df = spas_df.groupby(['Resort']).apply(lambda row: row.nlargest(5, 'rating')).reset_index(drop=True)

'''
'Shopping'
'''
# subset google on call categories and take top 5
shopping_df = mapping_google[mapping_google['Call Category']=='Shopping']

# get top 5 by rating for each resort
shopping_df = shopping_df.groupby(['Resort']).apply(lambda row: row.nlargest(5, 'rating')).reset_index(drop=True)


'''
'Medical'
'''
# subset google on call categories and take top 5
medical_df = mapping_google[mapping_google['Call Category']=='Medical']

# get top 5 by rating for each resort
medical_df = medical_df.groupby(['Resort']).apply(lambda row: row.nlargest(5, 'rating')).reset_index(drop=True)

'''
'Grocery'
'''
# subset google on call categories and take top 5
grocery_df = mapping_google[mapping_google['Call Category']=='Grocery']

# get top 5 by rating for each resort
grocery_df = grocery_df.groupby(['Resort']).apply(lambda row: row.nlargest(5, 'rating')).reset_index(drop=True)

'''
'Lodging'
'''
# subset google on call categories and take top 5
lodging_df = mapping_google[mapping_google['Call Category']=='Lodging']

# get top 5 by rating for each resort
lodging_df = lodging_df.groupby(['Resort']).apply(lambda row: row.nlargest(5, 'rating')).reset_index(drop=True)

'''
create mapping dictionaries
'''
# resorts
resort_dict = {'df': resort_complete_df,
               'color': 'blue',
               'radius_color': 'lightblue',
               'radius_col': 'Distance',
               'group': 'Resorts',
               'main': True}

# stations
station_dict = {'df': map_station_df,
                'color': 'black',
                'radius_color': None,
                'radius_col': None,
                'group': 'Stations',
                'main': False}

# google restaurants
restaurants_dict = {'df': restaurants_df,
                    'color': 'gray',
                    'radius_color': None,
                    'radius_col': None,
                    'group': 'Restaurants',
                    'main': False}

# google bars
bars_dict = {'df': bars_df,
             'color': 'darkblue',
             'radius_color': None,
             'radius_col': None,
             'group': 'Bars',
             'main': False}

# google spas
spas_dict = {'df': spas_df,
             'color': 'pink',
             'radius_color': None,
             'radius_col': None,
             'group': 'Spas',
             'main': False}

# google shopping
shopping_dict = {'df': shopping_df,
                 'color': 'lightgreen',
                 'radius_color': None,
                 'radius_col': None,
                 'group': 'Shopping',
                 'main': False}

# google medical
medical_dict = {'df': medical_df,
                'color': 'darkred',
                'radius_color': None,
                'radius_col': None,
                'group': 'Medical',
                'main': False}

# google grocery
grocery_dict = {'df': grocery_df,
                'color': 'darkgreen',
                'radius_color': None,
                'radius_col': None,
                'group': 'Grocery',
                'main': False}

# google lodging
lodging_dict = {'df':lodging_df,
                'color': 'purple',
                'radius_color': None,
                'radius_col': None,
                'group': 'Lodging',
                'main': False}

'''
create map_dict from other dictionaries:
    - resorts
    - stations
    - restaurants
    - bars
    - spas
    - shopping
    - medical
    - grocery
    - lodging
'''

map_df = pd.DataFrame([resort_dict,
                       station_dict,
                       restaurants_dict,
                       bars_dict,
                       spas_dict,
                       shopping_dict,
                       medical_dict,
                       grocery_dict,
                       lodging_dict])

'''
run the mapping function
'''
folium_map_complex(map_df, 'resort_map')
