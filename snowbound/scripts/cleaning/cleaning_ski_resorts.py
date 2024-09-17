# cleaning/cleaning_ski_resorts

'''
Module for cleaning ski resorts and combining data.

Data results from source/source_ski_resorts.py:
    - main raw data: ski_resorts_data.csv
    - epic raw data: epic_resorts.csv
    - ikon raw data: ikon_resorts.csv
    - raw regions us data: us_regions.csv
    - raw regions canada data: canada_regions_raw.csv

New data from:
    - kaggle: (https://www.kaggle.com/datasets/ulrikthygepedersen/ski-resorts?select=resorts.csv)
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
from fuzzywuzzy import process
import ast
import unicodedata

# custom module imports
sys.path.append('../source')
from source_google_functions import *

# get google api key
env_path = '../config/.env'
load_dotenv(env_path)
google_key = os.getenv('GOOGLE_KEY')


'''
Step 1: import data
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

# import data via function
relative_path = '../../data/'
resort_raw = import_data(relative_path, 'ski_resorts_data')
epic_raw = import_data(relative_path, 'epic_resorts')
ikon_raw = import_data(relative_path, 'ikon_resorts')
region_us_raw = import_data(relative_path, 'us_regions')
region_canada_raw = import_data(relative_path, 'canada_regions_raw')
open_close_raw = import_data(relative_path, 'kaggle_resorts')


'''
Step 2: save snippets of inital data
'''

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
    
# save snippets via function
save_path = '../../data/initial'
save_data_snippet(save_path, 'main_resort', resort_raw, head=True, head_size=10, snippet_index=[])
save_data_snippet(save_path, 'epic_resort', epic_raw, head=True, head_size=10, snippet_index=[])
save_data_snippet(save_path, 'ikon_resort', ikon_raw, head=True, head_size=10, snippet_index=[])
save_data_snippet(save_path, 'region_us', region_us_raw, head=True, head_size=10, snippet_index=[])
save_data_snippet(save_path, 'region_canada', region_canada_raw, head=True, head_size=10, snippet_index=[])
save_data_snippet(save_path, 'open_close', open_close_raw, head=True, head_size=10, snippet_index=[])


'''
Step 3: get coordinates via google api
'''

'''
# BLOCK COMMENTED OUT AFTER RUNNING TO REDUCE CALLS
# get raw addresses
addresses = resort_raw['Address'].to_list()

# run google api custom function get_coordinates
resort_coordinates = source_google_functions.get_coordinates(addresses, google_key)

# merge dataframe
resort_coordinates = pd.merge(resort_raw, resort_coordinates, on='Address')

# save dataframe
resort_coordinates.to_csv('../../data/resorts_with_coordinates.csv', index=False)
'''

# can load back in via function above
resort_coordinates = import_data(relative_path, 'resorts_with_coordinates')


'''
Step 4: find matches from epic and ikon resorts to resorts in resort_coordinates
'''

## EPIC ##
# add Epic as a column
epic_raw['Pass'] = 'Epic'

# remove Australia
epic_raw = epic_raw[epic_raw['Area'] != 'Australia']

# function to match resort names
def match_resorts(resort, choices, scorer):
    # result = process.extractOne(resort, choices, scorer=scorer)
    results = process.extract(resort, choices, scorer=scorer)
    if results[0][1] == 100:
        match = results[0][0]
    else:
        for num, result in enumerate(results):
            print(f'{resort} - {num} - {result[0]}')
        
        match_num = input('\nNumber of Matching Resort OR "n" for No Match\n')
        
        if match_num == "n":
            match = None
        else:
            match = results[int(match_num)][0]
    
    return match

'''
# BLOCK COMMENTED OUT AFTER RUNNING TO REDUCE CALLS
# run the match function
epic_raw['Match'] = epic_raw['Resort'].apply(lambda row: match_resorts(row, resort_coordinates['Resort'], process.fuzz.token_sort_ratio))
'''

'''
# BLOCK COMMENTED OUT AFTER RUNNING TO REDUCE CALLS
# save the matched results
epic_raw.to_csv('../../data/epic_resorts_matched.csv', index=False)
'''

# can load back in via function above
epic_raw = import_data(relative_path, 'epic_resorts_matched')

# prepare to merge
epic_match = epic_raw.copy()
epic_match.drop(columns=['Resort'], inplace=True)
epic_match.dropna(inplace=True)
epic_match = epic_match.rename(columns={'Area': 'Epic Area',
                                        'Match': 'Resort'})

## IKON ##
# add Ikon as column
ikon_raw['Pass'] = 'Ikon'

# constrain to US and Canada
ikon_raw = ikon_raw[(ikon_raw['Country'] == 'usa') | (ikon_raw['Country'] == 'canada')]

'''
# BLOCK COMMENTED OUT AFTER RUNNING TO REDUCE CALLS
# run the match function
ikon_raw['Match'] = ikon_raw['Resort'].apply(lambda row: match_resorts(row, resort_coordinates['Resort'], process.fuzz.token_sort_ratio))
'''

# noticed a few matches that weren't catching
ikon_raw[ikon_raw['Match'].isnull()]

# manual matches
ikon_manual = {'Big Bear Mountain Resort': 'Bear Mountain - Big Bear Lake',
               'Jackson Hole Mountain Resort': 'Jackson Hole',
               'Taos Ski Valley': 'Taos',
               'Solitude Mountain Resort': 'Solitude',
               'Alta Ski Area': 'Alta',
               'RED Mountain': 'Red Mountain Resort - Rossland'}

# apply manual matches
ikon_match = ikon_raw.copy()
for resort in ikon_manual:
    ikon_match.loc[ikon_match['Resort'] == resort, 'Match'] = ikon_manual[resort]
    
# blue mountain canada vs blue mountain usa
ikon_match.loc[ikon_match['Resort']=='Blue Mountain', 'Match'] = 'Blue Mountain Resort - Collingwood'

'''
# BLOCK COMMENTED OUT AFTER RUNNING TO REDUCE CALLS
# save the matched results
ikon_match.to_csv('../../data/ikon_resorts_matched.csv', index=False)
'''

# can load back in via function above
ikon_match = import_data(relative_path, 'ikon_resorts_matched')

# prepare to merge
ikon_match.drop(columns=['Resort', 'Country'], inplace=True)
ikon_match.dropna(inplace=True)
ikon_match = ikon_match.rename(columns={'Region': 'Ikon Area',
                                        'Match': 'Resort'})


'''
Step 5: merge pass dataframes onto main resort dataframe
'''
# rename pass columns
epic_match.rename(columns={'Pass': 'Epic_Pass'}, inplace=True)
ikon_match.rename(columns={'Pass': 'Ikon_Pass'}, inplace=True)

epic_ikon_df = pd.merge(epic_match, ikon_match, on='Resort', how='outer')

# combine pass columns
epic_ikon_df['Pass'] = epic_ikon_df['Epic_Pass'].combine_first(epic_ikon_df['Ikon_Pass'])

# drop individual pass columns
epic_ikon_df.drop(columns=['Epic_Pass', 'Ikon_Pass'], inplace=True)

# merge onto resorts
resort_passes = pd.merge(resort_coordinates, epic_ikon_df, how='left', on='Resort')

'''
Step 6: get proper addresses via google api
'''

'''
# BLOCK COMMENTED OUT AFTER RUNNING TO REDUCE CALLS
# run google api custom function get_proper_addresses
resort_proper = source_google_functions.get_proper_addresses(resort_passes, google_key)
resort_proper.to_csv('../../data/resorts_proper_addresses.csv', index=False)
'''

# can load back in via function above
resort_proper = import_data(relative_path, 'resorts_proper_addresses')

# what this result provides - save data snippet
save_data_snippet(save_path, 'resort_proper', resort_proper, head=True, head_size=10, snippet_index=[])


'''
Step 7: unpack address components from proper address results
'''

# function to unpack address components
def unpack_components(address_df):
    # list to store component results
    components = []
    # create copy of dataframe
    addresses = address_df.copy()
    # iterate through rows in dataframe
    for col, row in addresses.iterrows():
        resort = row['Resort']
        # saving and loading resort_proper can result in string type list element
        if type(row['Address Components']) == str:
            address_structure = ast.literal_eval(row['Address Components'])
        else:
            address_structure = row['Address Components']
            
        for component in address_structure:
            component['Resort'] = resort
            components.append(component)
        
    # unpack types
    for component in components:
        for num, locality_type in enumerate(component['types']):
            component[f'locality_type_{num}'] = locality_type
            
    # turn into dataframe
    components_df = pd.DataFrame(components)
    
    # drop types column
    components_df.drop(columns=['types'], inplace=True)
    
    # merge back proper address
    address_unpacked = pd.merge(components_df, address_df[['Resort', 'Proper Address']], on='Resort')
    
    return address_unpacked

# run function
address_unpacked = unpack_components(resort_proper)

# save result
address_unpacked.to_csv('../../data/addresses_unpacked.csv', index=False)

# what this result provides - save data snippet
save_data_snippet(save_path, 'address_unpacked', address_unpacked, head=True, head_size=10, snippet_index=[])


'''
Step 8: find errors in address_unpacked and cross reference with google places

- Concept: search local businesses until both states and cities are found
'''
# pivot on locality_type_0: contains administrative_area_level_1 (state) and locality (city)
unpacked_pivot = address_unpacked[['Resort', 'long_name', 'locality_type_0']].pivot(index='Resort', columns='locality_type_0', values='long_name').reset_index()

# find resorts missing state and city from unpacked_pivot
address_errors = resort_proper[resort_proper['Resort'].isin(unpacked_pivot[unpacked_pivot['locality'].isnull()]['Resort'])]

# bring in google unpacked (name unpack required only - see cleaning_google_places)
google_unpacked = import_data(relative_path, 'resorts_proper_addresses')

# function to cross reference locations with incompatible unpacked addresses - uses singular google api proper address
def error_cross_ref(google_unpacked, address_errors, google_key):
    # cross reference google unpacked with erroneous address unpacked results
    error_cross = google_unpacked[google_unpacked['resort'].isin(address_errors['Resort'])]
    
    # rename columns in error_cross for matching purposes
    error_cross.rename(columns={'latitude': 'Latitude',
                                'longitude': 'Longitude',
                                'resort': 'Resort'},
                       inplace=True)
    
    # resorts requiring a rerun
    rerun_resorts = list(error_cross['Resort'].unique())
    
    # rerun results storage
    rerun_results = []
    for resort in rerun_resorts:
        rerun_subset = error_cross[error_cross['Resort']==resort].reset_index(drop=True)
        for place in range(rerun_subset.shape[0]):
            place_subset = rerun_subset.iloc[place]
            resort = place_subset['Resort']
            latitude = place_subset['Latitude']
            longitude = place_subset['Longitude']
            place_address = source_google_functions.get_proper_address(resort, latitude, longitude, google_key)
            place_unpack = unpack_components(place_address)
            unpack_list = list(place_unpack['locality_type_0'].unique())
            
            # criteria for a successful rerun
            if ('administrative_area_level_1' in unpack_list) and ('locality' in unpack_list):
                rerun_results.append(place_unpack)
                break
    
    # concatenate results into single dataframe
    rerun_df = pd.concat(rerun_results, ignore_index=True)
    
    # return dataframe for results
    return rerun_df

'''
# BLOCK COMMENTED OUT AFTER RUNNING TO REDUCE CALLS
# run function
rerun_df = error_cross_ref(google_unpacked, address_errors, google_key)
'''


'''
Step 9: replace erroneous results with closest match
    - remove error_cross resorts from address_unpacked
    - restore resorts to address_unpacked using rerun_df
    - pivot
    - merge back in
'''

# remove error_cross resorts
rerun_resorts = list(address_errors['Resort'].unique())
unpack_complete = address_unpacked.copy()
unpack_complete = unpack_complete[~unpack_complete['Resort'].isin(rerun_resorts)]

# restore fixed error_cross resorts
unpack_complete = pd.concat([rerun_df, unpack_complete], ignore_index=True)

# pivot
complete_pivot = unpack_complete[['Resort', 'long_name', 'locality_type_0']].pivot(index='Resort', columns='locality_type_0', values='long_name').reset_index()

# drop any that weren't able to be matched
complete_pivot = complete_pivot[~complete_pivot['locality'].isnull()]
complete_pivot.reset_index(drop=True, inplace=True)

# drop columns containing null values
complete_pivot.dropna(axis=1, how='any', inplace=True)

# merge back into main df
resort_locality = pd.merge(complete_pivot, resort_passes, on='Resort')
resort_locality.to_csv('../../data/resort_locality_dirty.csv', index=False)


'''
Step 10: clean resort names, administrative_level_1, and locality
    - remove temporarily closed (the data of a closed resort could still be useful for mountainous models)
    - normalize text
'''

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

# apply to resort_locality across required fields
resort_locality_cleaned = resort_locality.copy()
resort_locality_cleaned['Resort'] = resort_locality_cleaned['Resort'].apply(clean_text_v1)
resort_locality_cleaned['administrative_area_level_1'] = resort_locality_cleaned['administrative_area_level_1'].apply(clean_text_v1)
resort_locality_cleaned['locality'] = resort_locality_cleaned['locality'].apply(clean_text_v1)


'''
Step 11: drop additional columns and rows
    - remove any stated indoor resort rows (doesn't help with mountainous models)
    - remove columns with unrecoverable data
'''

# remove indoor resorts
resort_locality_cleaned = resort_locality_cleaned[~resort_locality_cleaned['Resort'].str.contains('indoor ski area')]
resort_locality_cleaned.reset_index(drop=True, inplace=True)

'''
drop the following columns:
    - Region: only has North America
    - Country: lowercase country has longname
    - Locale 1: administrative_area_level_1 & locality better contain a better summary
    - Locale 2: administrative_area_level_1 & locality better contain a better summary
    - Link: only required for scraping purposes
    - Address: messy and administrative_area_level_1 & locality better contain a better summary
    - Snow reliability: over 84% missing values
    - Slope preparation: over 84% missing values
    - Access, on-site parking: over 84% missing values
    - Orientation (trail map, information boards, sign-postings): over 84% missing values
    - Cleanliness and hygiene: over 84% missing values
    - Environmentally friendly ski operation: over 84% missing values
    - Friendliness of staff: over 84% missing values
    - Mountain restaurants, ski huts, gastronomy: over 84% missing values
    - Après-ski: over 84% missing values
    - Accommodation offering directly at the slopes and lifts: over 84% missing values
    - Families and children: over 84% missing values
    - Beginners: over 84% missing values
    - Advanced skiers, freeriders: over 84% missing values
    - Snow parks: over 84% missing values
    - Cross-country skiing and trails: over 84% missing values
    - Epic Area: administrative_area_level_1 & locality better contain a better summary
    - Ikon Area: administrative_area_level_1 & locality better contain a better summary
'''

drop_columns = ['Region', 'Country', 'Locale 1', 'Locale 2', 'Link', 'Address', 'Snow reliability',
                'Slope preparation', 'Access, on-site parking', 'Orientation (trail map, information boards, sign-postings)',
                'Cleanliness and hygiene', 'Environmentally friendly ski operation', 'Friendliness of staff',
                'Mountain restaurants, ski huts, gastronomy', 'Après-ski', 'Accommodation offering directly at the slopes and lifts',
                'Families and children', 'Beginners', 'Advanced skiers, freeriders', 'Snow parks',
                'Cross-country skiing and trails', 'Epic Area', 'Ikon Area']

resort_locality_cleaned.drop(columns=drop_columns, inplace=True)

# check what's left
resort_locality_cleaned.isnull().sum()
'''
Results:
    
Resort                               0
administrative_area_level_1          0
country                              0
locality                             0
Overall Rating                       0
Elevation Difference                 0
Elevation Low                        0
Elevation High                       0
Trails Total                         0
Trails Easy                          0
Trails Intermediate                  0
Trails Difficult                     0
Lifts                                0
Price                               33
Ski resort size                      0
Slope offering, variety of runs      0
Lifts and cable cars                 0
Latitude                             0
Longitude                            0
Pass                               311

Analysis:
- Price:
    - missing: 8.7%
    - decision: given a high correlation between price and overall rating, fill with mean price of close ratings
- Pass:
    - missing: 82.1%
    - decision: likely means another type of pass or independent, replace with 'Other'
    
'''

# fill null pass types with Other
resort_locality_cleaned['Pass'] = resort_locality_cleaned['Pass'].fillna('Other')

# confirm high correlation - 0.86
price_rating_corr = resort_locality_cleaned['Price'].corr(resort_locality_cleaned['Overall Rating'])

# ratings of missing prices
resort_missing_prices = resort_locality_cleaned[resort_locality_cleaned['Price'].isnull()][['Resort', 'Overall Rating']].reset_index(drop=True)

# average price of that (or nearest rating) in non-null price dataset
resort_eligible_prices = resort_locality_cleaned[resort_locality_cleaned['Price'].notnull()].reset_index(drop=True)

# function to find the average price near given ratings
def average_near_rating(missing, eligible, tolerance=0.1):
    # resort - price dictionary
    average_price = dict()
    
    # parse through missing dataframe rows
    for index, row in missing.iterrows():
        # get information
        resort = row['Resort']
        rating = row['Overall Rating']
        
        # find average
        eligible_subset = eligible[(eligible['Overall Rating'] >= rating - tolerance) & 
                                   (eligible['Overall Rating'] <= rating + tolerance)]
        
        if not eligible_subset.empty:
            eligible_avg = eligible_subset['Price'].mean()
        else:
            eligible_avg = None
        
        # populate dictionary
        average_price[resort] = int(eligible_avg)
    
    # return
    return average_price

# run function        
new_prices = average_near_rating(resort_missing_prices, resort_eligible_prices, tolerance=0.1)

# populate resorts with the new prices
resort_locality_cleaned.loc[resort_locality_cleaned['Price'].isnull(), 'Price'] = resort_locality_cleaned['Resort'].map(new_prices)


'''
Step 12: Add Official US and Canada Regions
    - alter canada region dataframe
    - merge us and canada into resort_locality_cleaned
'''

## CANADA ##
# drop rank and capital
region_canada_raw.drop(columns=['Rank', 'Capital City'], inplace=True)

# change name to just province_territory
region_canada_raw.rename(columns={'Province/Territory': 'province_territory'}, inplace=True)

# comma needed between Nunavut and Northwest Territories
region_canada_raw.loc[region_canada_raw['Region']=='North', 'province_territory'] = region_canada_raw.loc[region_canada_raw['Region']=='North', 'province_territory'].str.replace('\r\n', ',')

# split province by comma into own rows
# first split column values into a list
region_canada_raw['province_territory'] = region_canada_raw['province_territory'].str.split(',')

# explode the columns
region_canada_raw = region_canada_raw.explode('province_territory').reset_index(drop=True)

# apply strip
region_canada_raw['province_territory'] = region_canada_raw['province_territory'].str.strip()

# remove rows with blank values (additional )
region_canada_raw = region_canada_raw[region_canada_raw['province_territory'] != '']
region_canada_raw.reset_index(drop=True, inplace=True)

# save cleaned dataframe
region_canada_raw.to_csv('../../data/canada_regions.csv', index=False)

## MERGE ##
# copy resorts
resorts = resort_locality_cleaned.copy()

# split by country
resorts_us = resorts[resorts['country'] == 'United States']['administrative_area_level_1'].unique()
resorts_canada = resorts[resorts['country'] == 'Canada']['administrative_area_level_1'].unique()

# find mismatches
mismatches_us = [area for area in resorts_us if area not in region_us_raw['State Name'].unique()]
print(mismatches_us)
mismatches_canada = [area for area in resorts_canada if area not in region_canada_raw['province_territory'].unique()]
print(mismatches_canada)

# rename Yukon Territory to Yukon
region_canada_raw['province_territory'].replace({'Yukon Territory': 'Yukon'}, inplace=True)

# in resorts, change Nouveau-Brunswick to New Brunswick
resorts['administrative_area_level_1'].replace({'Nouveau-Brunswick': 'New Brunswick'}, inplace=True)

# merge us and canada on
us_merge = pd.merge(resorts[resorts['country']=='United States'],
                    region_us_raw,
                    left_on='administrative_area_level_1',
                    right_on='State Name')

canada_merge = pd.merge(resorts[resorts['country']=='Canada'],
                        region_canada_raw,
                        left_on='administrative_area_level_1',
                        right_on='province_territory')

# concatenate merges
resort_new = pd.concat([us_merge, canada_merge])

# check results - regions unique by country
resort_new.groupby(['country'])['Region'].value_counts()

# check results - total value counts is equal to unique rows
resort_new.groupby(['country'])['Region'].value_counts().sum()

# drop State Name and province_territory
resort_new.drop(columns=['State Name', 'province_territory', 'Abbreviation'], inplace=True)

# check results - null values
resort_new.isnull().sum()

# rename columns
resort_col_rename = {'administrative_area_level_1': 'state_province_territory',
                     'country': 'Country',
                     'locality': 'City',
                     'Ski resort size': 'Resort Size',
                     'Slope offering, variety of runs': 'Run Variety',
                     'Lifts and cable cars': 'Lifts Quality'}

# create copy
resort_final = resort_new.copy()

# apply rename
resort_final.rename(columns=resort_col_rename, inplace=True)

# save dataframe
resort_final.to_csv('../../data/resort_cleaned.csv', index=False)

# save snippets
save_path = '../../data/cleaned'
save_data_snippet(save_path, 'resorts', resort_final, head=True, head_size=10, snippet_index=[])
