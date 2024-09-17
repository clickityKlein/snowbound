# source/source_weather

'''
Module for gathering weather data and associated stations data surrounding ski resorts.

Module requires the following:
    - from the cleaning_ski_resorts: recreate import_data function
'''

# library imports
import requests
import pandas as pd
from dotenv import load_dotenv
import os

# get visualcrossing api key
env_path = '../config/.env'
load_dotenv(env_path)
vc_key = os.getenv('VC_KEY')


'''
Function created in cleaning_ski_resorts
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

# import resorts with coordinates
relative_path = '../../data/'
resort_coordinates = import_data(relative_path, 'resorts_with_coordinates')


'''
Functions to call the API for locations surrounding ski resorts
'''

# function to fetch weather data from vc
def get_weather_data(resort, latitude, longitude, start_date, end_date, vc_key):
    # build url
    base_url = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/'
    built_url = f'{base_url}{latitude}%2C%20{longitude}/{start_date}/{end_date}?unitGroup=us&include=days&key={vc_key}&contentType=json'
    # request data
    response = requests.request('GET', built_url)
    # initiate data storage
    weather = []
    stations = []
    # status code error handling
    if response.status_code != 200:
        print('Unexpected Status Code: ', response.status_code)
    else:
        data = response.json()
        if 'days' in data:
            for weather_result in data['days']:
                weather_result['resort'] = resort
                weather.append(weather_result)
        else:
            print(f'Weather Data ("days") Ineligible for {resort}')
        if 'stations' in data:
            for station_result in data['stations']:
                data['stations'][station_result]['resort'] = resort
                stations.append(data['stations'][station_result])
        else:
            print(f'Stations Data ("stations") Ineligible for {resort}')
            
    return weather, stations

# function to extend get_weather_data to multiple stations
def get_weather_data_complete(resort_df, start_date, end_date, vc_key):
    weather_complete = []
    stations_complete = []
    for col, row in resort_df.iterrows():
        resort = row['Resort']
        latitude = row['Latitude']
        longitude = row['Longitude']
        weather, stations = get_weather_data(resort, latitude, longitude, start_date, end_date, vc_key)
        weather_complete.extend(weather)
        stations_complete.extend(stations)
        
        # progress report
        resort_index = resort_df[resort_df['Resort']==resort].index[0]
        completion = ((resort_index + 1) / resort_df.shape[0]) * 100
        print(f'{completion:.2f}% Complete - {resort}')
    
    return weather_complete, stations_complete


'''
run functions
'''

'''
# BLOCK COMMENTED OUT AFTER RUNNING TO REDUCE CALLS
# gather data
weather_complete, stations_complete = get_weather_data_complete(resort_coordinates, '2019-01-01', '2024-08-31', vc_key)
'''

# turn into dataframes
weather_df = pd.DataFrame(weather_complete)
stations_df = pd.DataFrame(stations_complete)

# save data
weather_df.to_csv('../../data/vc_weather.csv', index=False)
stations_df.to_csv('../../data/vc_stations.csv', index=False)
