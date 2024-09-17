# cleaning/cleaning_weather

'''
Module to clean weather data and apply station data

Data results from source/source_weather.py:
    - weather: vc_weather.csv
    - stations: vc_stations.csv
    
Module requires the following:
    - from the cleaning_ski_resorts: recreate import_data function, save_data_snippet function, clean_text_v1
'''

# library imports
import pandas as pd
import re
from geopy.distance import geodesic
import requests
import json
import time
import sys
import os
import ast
import unicodedata
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer


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

# function to save snippet of data
def save_data_snippet(save_path, save_name, df, head=True, head_size=10, snippet_index=[]):
    # snippet index for problem areas
    # create save file path
    file_path_null = os.path.join(save_path, f'{save_name}_null.csv')
    file_path_snippet = os.path.join(save_path, f'{save_name}_snippet.csv')
    
    # create copy
    df_snippet = df.copy()
    
    # find columns with null values
    df_null = df_snippet.isnull().sum().reset_index()
    df_null.columns = ['Column', 'Null Count']
    
    # get snippet
    if head:
        df_snippet = df_snippet.head(head_size)
    elif len(snippet_index) > 0:
        df_snippet = df_snippet.iloc[snippet_index]
    else:
        print('Function Set Up Incorrect')
        return None
    
    # save null value dataframe and snippet dataframe
    df_null.to_csv(file_path_null, index=False)
    df_snippet.to_csv(file_path_snippet, index=False)
    
# function to clean text for across multiple facets
def clean_text_v1(text):
    # remove temporarily closed
    if '(temporarily closed)' in text:
        text = text.replace('(temporarily closed)', '')
        
    # change curly apostrophes to straight
    text = text.replace("’", "'")
    
    # change accent apostrophe to straight
    text = text.replace("`", "'")
    
    # normalize accent characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    
    # strip text
    text = text.strip()
    
    # remove double-plus spaces between words
    text = " ".join(text.split())
    
    return text


'''
import weather data and create snippets
'''

# can load back in via function above
relative_path = '../../data/'
weather = import_data(relative_path, 'vc_weather')
stations = import_data(relative_path, 'vc_stations')

# save snippets
save_path = '../../data/initial'
save_data_snippet(save_path, 'weather', weather, head=True, head_size=10, snippet_index=[])
save_data_snippet(save_path, 'stations', stations, head=True, head_size=10, snippet_index=[])


'''
ABOUT

Visual Crossing's Metrics:
    - Temperature: Degrees Fahrenheit
    - Heat Index: Degrees Fahrenheit
    - Wind Chill: Degrees Fahrenheit
    - Precipitation: Inches
    - Snow: Inches
    - Wind: Miles Per Hour
    - Wind Gust: Miles Per Hour
    - Visibility: Miles
    - Pressure: Millibars
    - Solar Radiation: W/m^2
    - Solar Energy: MJ/m^2
    - Soil Moisture: Inches
    - Station Distance: Meters
    
Visual Crossing's Data Descriptions:
    - cloudcover – how much of the sky is covered in cloud ranging from 0-100%
    - conditions – textual representation of the weather conditions. See Weather Data Conditions.
    - description – longer text descriptions suitable for displaying in weather displays. The descriptions combine the main features of the weather for the day such as precipitation or amount of cloud cover. Daily descriptions are provided for historical and forecast days. When the timeline request includes the model forecast period, a seven day outlook description is provided at the root response level.
    - datetime – ISO 8601 formatted date, time or datetime value indicating the date and time of the weather data in the local time zone of the requested location. See Dates and Times in the Weather API for more information.
    - datetimeEpoch – number of seconds since 1st January 1970 in UTC time
    - tzoffset – the time zone offset in hours. This will only occur in the data object if it is different from the global time zone offset.
    - dew – dew point temperature
    - feelslike – what the temperature feels like accounting for heat index or wind chill. Daily values are average values (mean) for the day.
    - feelslikemax (day only) – maximum feels like temperature at the location.
    - feelslikemin (day only) – minimum feels like temperature at the location.
    - hours – array of hourly weather data objects. This is a child of each of the daily weather object when hours are selected.
    - humidity – relative humidity in %
    - icon – a fixed, machine readable summary that can be used to display an icon
    - moonphase –  represents the fractional portion through the current moon lunation cycle ranging from 0 (the new moon) to 0.5 (the full moon) and back to 1 (the next new moon). See How to include sunrise, sunset, moon phase, moonrise and moonset data into your API requests
    - normal – array of normal weather data values – Each weather data normal is an array of three values representing, in order, the minimum value over the statistical period, the mean value, and the maximum value over the statistical period.
    - offsetseconds (hourly only) – time zone offset for this weather data object in seconds – This value may change for a location based on daylight saving time observation.
    - precip – the amount of liquid precipitation that fell or is predicted to fall in the period. This includes the liquid-equivalent amount of any frozen precipitation such as snow or ice.
    - precipcover (days only) – the proportion of hours where there was non-zero precipitation
    - precipprob (forecast only) – the likelihood of measurable precipitation ranging from 0% to 100%
    - preciptype – an array indicating the type(s) of precipitation expected or that occurred. Possible values include rain, snow, freezingrain and ice.
    - pressure – the sea level atmospheric or barometric pressure in millibars (or hectopascals)
    - snow – the amount of snow that fell or is predicted to fall
    - snowdepth – the depth of snow on the ground
    - source –  the type of weather data used for this weather object. – Values include historical observation (“obs”), forecast (“fcst”), historical forecast (“histfcst”) or statistical forecast (“stats”). If multiple types are used in the same day, “comb” is used. Today a combination of historical observations and forecast data.
    - stations (historical only) – the weather stations used when collecting an historical observation record
    - sunrise (day only) – The formatted time of the sunrise (For example “2022-05-23T05:50:40”). See How to include sunrise, sunset, moon phase, moonrise and moonset data into your API requests
    - sunriseEpoch – sunrise time specified as number of seconds since 1st January 1970 in UTC time
    - sunset – The formatted time of the sunset (For example “2022-05-23T20:22:29”). See How to include sunrise, sunset, moon phase, moonrise and moonset data into your API requests
    - sunsetEpoch – sunset time specified as number of seconds since 1st January 1970 in UTC time
    - moonrise (day only, optional) – The formatted time of the moonrise (For example “2022-05-23T02:38:10”). See How to include sunrise, sunset, moon phase, moonrise and moonset data into your API requests
    - moonriseEpoch (day only, optional) – moonrise time specified as number of seconds since 1st January 1970 in UTC time
    - moonset (day only, optional) – The formatted time of the moonset (For example “2022-05-23T13:40:07”)
    - moonsetEpoch (day only, optional) – moonset time specified as number of seconds since 1st January 1970 in UTC time
    - temp – temperature at the location. Daily values are average values (mean) for the day.
    - tempmax (day only) – maximum temperature at the location.
    - tempmin (day only) – minimum temperature at the location.
    - uvindex – a value between 0 and 10 indicating the level of ultra violet (UV) exposure for that hour or day. 10 represents high level of exposure, and 0 represents no exposure. The UV index is calculated based on amount of short wave solar radiation which in turn is a level the cloudiness, type of cloud, time of day, time of year and location altitude. Daily values represent the maximum value of the hourly values.
    - uvindex2 (optional, 5 day forecast only) – an alternative UV index element that uses the algorithms and models used by the US National Weather Service. In order to maintain backwards compatibility, this UV index element is deployed as a new, optional element ‘uvindex2’ and may be requested using the elements parameter.
    - visibility – distance at which distant objects are visible
    - winddir – direction from which the wind is blowing
    - windgust – instantaneous wind speed at a location – May be empty if it is not significantly higher than the wind speed. Daily values are the maximum hourly value for the day.
    - windspeed – the sustained wind speed measured as the average windspeed that occurs during the preceding one to two minutes. Daily values are the maximum hourly value for the day.
    - windspeedmax (day only, optional) – maximum wind speed over the day.
    - windspeedmean (day only , optional ) – average (mean) wind speed over the day.
    - windspeedmin (day only , optional ) – minimum wind speed over the day.
    - solarradiation – (W/m2) the solar radiation power at the instantaneous moment of the observation (or forecast prediction). See the full solar radiation data documentation and Wind and Solar Energy pages.
    - solarenergy – (MJ /m2 ) indicates the total energy from the sun that builds up over an hour or day. See the full solar radiation data documentation and Wind and Solar Energy pages .
    - severerisk (forecast only) – a value between 0 and 100 representing the risk of convective storms (e.g. thunderstorms, hail and tornadoes). Severe risk is a scaled measure that combines a variety of other fields such as the convective available potential energy (CAPE) and convective inhibition (CIN), predicted rain and wind. Typically, a severe risk value less than 30 indicates a low risk, between 30 and 70 a moderate risk and above 70 a high risk.
'''


'''
Step 1: clean resort and station names
'''

# create copies
weather_df = weather.copy()
stations_df = stations.copy()

# apply resort name cleaning
weather_df['resort'] = weather_df['resort'].apply(clean_text_v1)
stations_df['resort'] = stations_df['resort'].apply(clean_text_v1)

'''
Step 2: null values and column importance
'''

## STATIONS ##
# useCount
stations_df['useCount'].value_counts()
'''
0    7128

# we can drop
'''

# contribution
stations_df['contribution'].value_counts()
'''
0.0    7128

# we can drop
'''

# quality likely won't be applicable to this project
stations_df.drop(columns=['useCount', 'contribution', 'quality'], inplace=True)

# save cleaned stations
stations_df.to_csv('../../data/stations_cleaned.csv', index=False)

## WEATHER ##
weather_df.isnull().sum()
weather_df['preciptype'].value_counts()
weather_df['source'].value_counts()
weather_df['icon'].value_counts()
weather_df['conditions'].value_counts()
weather_df['description'].value_counts()
weather_df.isnull().sum()
weather_df['visibility'].describe()
weather_df['severerisk'].describe()

# icon on preciptype null days and non-null dayas
weather_df[weather_df['preciptype'].isnull()]['icon'].value_counts()
'''
partly-cloudy-day    154002
clear-day            140311
cloudy                 8066
wind                   2807
fog                      83
'''

weather_df[weather_df['preciptype'].notnull()]['icon'].value_counts()
'''
rain                 234099
snow                 137150
partly-cloudy-day     79627
clear-day             19413
cloudy                11379
wind                   2002
fog                      96
'''

'''
conclusion of condition-combination:
    - preciptype with icon is likely a good condition combination
    - when preciptype is null, we're missing rain and snow

drop:
    - datetimeEpoch: not applicable to this project
    - sunriseEpoch: not applicable to this project
    - sunsetEpoch: not applicable to this project
    - conditions: icon + preciptype is sufficient
    - description: icon + preciptype is sufficient
    - source: all observations
'''
drop_columns = ['datetimeEpoch', 'sunriseEpoch', 'sunsetEpoch', 'conditions', 'description', 'source']
weather_df.drop(columns=drop_columns, inplace=True)

# revisit null values
weather_df.isnull().sum()
'''
fix nulls:
    - preciptype: that likely means no precipitation for that day, replace with list like ['none']
    - windgust: take the average windgust for that location around that windspeed
    - visibility: take the average visibility for that location at that icon
    - solarradiation, solarenergy, uvindex: take the average of each for that location around that cloudcover
    - tzoffset: null means the data object is not different from the global time zone offset: replace with 0
    - severerisk: even though this is supposed to be a forecast only variable, there are suitable values, however they are missing the 0 value, so assuming a replacement of 0 is ideal

check for outliers on numeric null columns before applying null average fixes:
    - windgust
    - visibiltiy
    - solarradation
    - solarenergy
    - unvindex
    - tzoffset
    - severerisk
'''

# check for erroroneous outliers before applying averaging methods to fix null values
base_color = sns.color_palette()[0]
outlier_nulls = ['windgust', 'visibility', 'solarradiation', 'solarenergy', 'uvindex', 'tzoffset', 'severerisk']
outlier_melted = pd.melt(weather_df[outlier_nulls])
plt.figure(figsize=(12, 8))
sns.boxplot(x='value', y='variable', data=outlier_melted, color=base_color)
plt.xlabel('Observation')
plt.ylabel('Column')
plt.title('Outlier Detection')
plt.savefig('weather_prenull_outliers', dpi=300)
plt.show()

'''
Include information about normal ranges of these values:
    - looks like visibility definitely has an error (normally there's a max of 150)
    - we'll set this value to null and apply our function to it as well
'''

# set values above 200 miles to NoneType
weather_df.loc[weather_df['visibility'] > 200, 'visibility'] = None

# first turn datetime column into datetime type
weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])

# function to fill missing values with subsetted averages
def groupby_averages(df_unclean, missing_value):
    # create a copy
    df = df_unclean.copy()
    
    # create a month column
    df['month'] = df['datetime'].dt.month
    
    # calculate average by resort and month
    df['average'] = df.groupby(['resort', 'month'])[missing_value].transform('mean')
    
    # fill values
    df[missing_value].fillna(df['average'], inplace=True)
    
    # drop temorary columns
    df.drop(columns=['month', 'average'], inplace=True)
    
    # return 
    return df
    

## RUN FUNCTIONS ##
# copy
weather_cleaned = weather_df.copy()

# windgust
weather_cleaned = groupby_averages(weather_cleaned, 'windgust')

# visibility
weather_cleaned = groupby_averages(weather_cleaned, 'visibility')

# solarradiation
weather_cleaned = groupby_averages(weather_cleaned, 'solarradiation')

# solarenergy
weather_cleaned = groupby_averages(weather_cleaned, 'solarenergy')

# uvindex
weather_cleaned = groupby_averages(weather_cleaned, 'uvindex')


## VALUES NOT REQUIRING AVERAGING METHODS ##
# when is preciptype null that is equivalent to ['none']
weather_cleaned['preciptype'] = weather_cleaned['preciptype'].apply(lambda row: ['none'] if pd.isnull(row) else row)

# when tzoffset is null that is equivalent to 0
weather_cleaned['tzoffset'].fillna(0, inplace=True)

# when severerisk is null that is equivalent to 0
weather_cleaned['severerisk'].fillna(0, inplace=True)


'''
unpacking lists:
    - preciptype
    - station
'''

## UNPACK PRECIPTYPE ##
# function to convert column to all list types
def convert_to_list(item):
    if type(item) != list:
        return ast.literal_eval(item)
    else:
        return item

weather_cleaned['preciptype'] = weather_cleaned['preciptype'].apply(convert_to_list)

# multilabel binarizer initialization and application
mlb = MultiLabelBinarizer()
encoded_data = mlb.fit_transform(weather_cleaned['preciptype'])
encoded_df = pd.DataFrame(encoded_data, columns=mlb.classes_)

# rename new columns for preciptypes encoded
rename_cols = {'freezingrain': 'type_freezingrain',
               'ice': 'type_ice',
               'none': 'type_none',
               'rain': 'type_rain',
               'snow': 'type_snow'}

encoded_df.rename(columns=rename_cols, inplace=True)

# bring back into the dataframe
weather_cleaned = pd.concat([weather_cleaned, encoded_df], axis=1)

# drop column used in encoding
weather_cleaned.drop(columns=['preciptype'], inplace=True)


## UNPACK STATION ##
'''
We'll use a slightly different approaach which puts the result in a transaction or basket data type format
'''
# function to turn stations into a basket type dataframe format
def stations_to_basket(df, list_col):
    # create copy
    df = df.copy()
    # get maximum length of any list in the dataframe column
    max_items = df[list_col].apply(len).max()
    stations_basket = {'resort': [], 'datetime': []}
    for station in range(max_items):
        stations_basket[f'station_{station}']=[]
    
    # loop and fill
    for index, row in df.iterrows():
        # get components
        resort = row['resort']
        date = row['datetime']
        stations = row['stations']
        
        # fill dictionary
        stations_basket['resort'].append(resort)
        stations_basket['datetime'].append(date)
        for station in range(max_items):
            try:
                stations_basket[f'station_{station}'].append(stations[station])
            except:
                stations_basket[f'station_{station}'].append(None)
    
    # return dataframe
    return pd.DataFrame(stations_basket)

# apply to list function to account for string encoded lists
weather_cleaned['stations'] = weather_cleaned['stations'].apply(convert_to_list)

# save weather cleaned
weather_cleaned.to_csv('../../data/weather_cleaned.csv', index=False)

# save snippets
save_path = '../../data/cleaned'
save_data_snippet(save_path, 'weather', weather_cleaned, head=True, head_size=10, snippet_index=[])

# apply basket function
weather_stations_basket = stations_to_basket(weather_cleaned, 'stations')

# save the basket
weather_stations_basket.to_csv('../../data/stations_basket.csv', index=False)
