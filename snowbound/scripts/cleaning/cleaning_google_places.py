# cleaning/cleaning_google_places

'''
Module for cleaning results from Google Places API calls.

Data results from source/source_google_places.py:
    - google places: google_df.csv
    
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
import emoji


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


'''
import places data and create snippets
'''

# can load back in via function above
relative_path = '../../data/'
google_df = import_data(relative_path, 'google_places')

# save snippets
save_path = '../../data/initial'
save_data_snippet(save_path, 'google_places', google_df, head=True, head_size=10, snippet_index=[])

# problem areas in google_df
problem_index = [24892, 16893, 2563, 16894, 8371, 27806, 11680, 25373, 26648, 20597]
save_data_snippet(save_path, 'google_places_problematic', google_df, head=False, snippet_index=problem_index)


'''
unpack google places information - name
'''

# function to unpack google places data - initial
def unpack_google_places(google_df):
    # create copy of initial google places dataframe
    df = google_df.copy()
    
    # data storage for unpacked places
    google_unpacked = {'latitude': [],
                       'longitude': [],
                       'name': [],
                       'rating': [],
                       'types': [],
                       'total_ratings': [],
                       'vicinity': [],
                       'resort': [],
                       'call_category': [],
                       'price_level': []}
    
    # iterate through dataframe
    for col, row in df.iterrows():
        # dictionary from geometry column could've been reverted back to string type
        if type(row['geometry']) != dict:
            coord_dict = ast.literal_eval(row['geometry'])
            
        # append information into data storage
        google_unpacked['latitude'].append(coord_dict['location']['lat'])
        google_unpacked['longitude'].append(coord_dict['location']['lng'])
        google_unpacked['name'].append(row['name'])
        google_unpacked['rating'].append(row['rating'])
        google_unpacked['types'].append(row['types'])
        google_unpacked['total_ratings'].append(row['user_ratings_total'])
        google_unpacked['vicinity'].append(row['vicinity'])
        google_unpacked['resort'].append(row['resort'])
        google_unpacked['call_category'].append(row['category'])
        google_unpacked['price_level'].append(row['price_level'])
        
    # return dataframe
    return pd.DataFrame(google_unpacked)

# run function
google_unpacked = unpack_google_places(google_df)

# save results
google_unpacked.to_csv('../../data/google_unpacked.csv', index=False)


'''
unpack google places information - type
'''

# function to clean text for across multiple facets - from cleaning_ski_resorts.py
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

# create copy and clean text
google_cleaned = google_unpacked.copy()
google_cleaned['resort'] = google_cleaned['resort'].apply(clean_text_v1)

# function to unpack google places data - initial
def unpack_google_types(google_df):
    df = google_df.copy()
    google_types = []
    for col, row in df.iterrows():
        categories = row['types']
        
        if type(categories) == str:
            categories = ast.literal_eval(categories)
            
        resort = row['resort']
        call_category = row['call_category']
        name = row['name']
        latitude = row['latitude']
        longitude = row['longitude']
        
        type_dict = {'Resort': resort,
                     'Call Category': call_category,
                     'Name': name,
                     'Latitude': latitude,
                     'Longitude': longitude}
        
        for num, category in enumerate(categories):
            type_dict[f'category_{num}'] = category
        
        google_types.append(type_dict)
        
    return pd.DataFrame(google_types)

# run function
google_types = unpack_google_types(google_cleaned)

# save google_types (could be eligible for ARM)
google_types.to_csv('../../data/google_types.csv', index=False)

# save snippets
save_path = '../../data/initial'
save_data_snippet(save_path, 'google_types', google_types, head=True, head_size=10, snippet_index=[])

# look at type results
google_types.isnull().sum()

# category_0 - category_2 have non-null values <-> drop the rest
google_types.dropna(axis=1, how='any', inplace=True)

# rename categories
google_types.rename(columns={'category_0': 'Initial Category',
                             'category_1': 'Secondary Category',
                             'category_2': 'Tertiary Category'},
                    inplace=True)

# rename columnns in google_cleaned
google_cleaned.rename(columns={'name': 'Name',
                               'resort': 'Resort',
                               'latitude': 'Latitude',
                               'longitude': 'Longitude',
                               'call_category': 'Call Category'},
                      inplace=True)

# merge unpacked categories back into google_cleaned
google_merged = pd.merge(google_cleaned, google_types, on=['Resort', 'Call Category', 'Name', 'Latitude', 'Longitude'])

# duplicates may have arisen due to resorts in close proximity - drop businesses that have identical names and coordinates
google_merged.drop_duplicates(subset=['Latitude', 'Longitude', 'Name'], inplace=True)

# check null values
google_merged.isnull().sum()
'''
Latitude                  0
Longitude                 0
Name                      0
rating                 3219
types                     0
total_ratings          3219
vicinity                 19
Resort                    0
Call Category             0
price_level           18234
Initial Category          0
Secondary Category        0
Tertiary Category         0
'''

'''
Column Cleaning:
    - types: deprecated by initial, secondary, and tertiary (drop)
    - vicinity: lat and long are superior (drop)
    - price_level: 79.6% missing (drop)
    - rating & total_rating: only null due to no ratings yet (fill nulls with 0s)
'''

# drop types and vicinity
google_merged.drop(columns=['types', 'vicinity', 'price_level'], inplace=True)

# fill null rating and total_rating
google_merged.fillna(0, inplace=True)


'''
clean google places names:
    - remove emojis
    - normalize text
    - combine into a single function
'''

# create copy
google_clean_names = google_merged.copy()

# function to remove emojis
def remove_emoji(text):
    return emoji.replace_emoji(text, replace='')

# function to normalize text
def normalize_text(text):
    # change curly apostrophes to straight
    text = text.replace("’", "'")
    
    # change accent apostrophe to straight
    text = text.replace("`", "'")
    
    # normalize accent characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    
    # strip names
    try:
        text = text.strip()
        text = ast.literal_eval(text).strip()
        return text
    except:
        return text.strip()
    
# function to combine emoji removal and normalize text
def clean_text_v2(text):
    text = remove_emoji(text)
    text = normalize_text(text)
    
    return text

# run function
google_clean_names['Name'] = google_clean_names['Name'].apply(clean_text_v2)

# remove blank names
google_clean_names = google_clean_names[google_clean_names['Name'] != '']

# remove names with only a single character
google_clean_names = google_clean_names[google_clean_names['Name'].str.len() > 1]

# reset index
google_clean_names.reset_index(drop=True, inplace=True)

# save cleaned google business dataframe
google_clean_names.to_csv('../../data/google_cleaned.csv', index=False)

# save snippets
save_path = '../../data/cleaned'
save_data_snippet(save_path, 'google_places', google_clean_names, head=True, head_size=10, snippet_index=[])
