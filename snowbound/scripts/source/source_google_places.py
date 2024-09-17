# source/source_google_functions

'''
Module for gathering Google Places data at resorts.

Module requires the following:
    - source_google_functions: import fetch_google_places_data function
    - from cleaning_ski_resorts process: import resorts_with_coordinates dataframe
    - from the cleaning_ski_resorts: recreate import_data function and save_data_snippet function
'''

# library imports
import pandas as pd
import sys
import os
from dotenv import load_dotenv

# custom module imports - source folder
sys.path.append('../source')
from source_google_functions import *

# get google api key
env_path = '../config/.env'
load_dotenv(env_path)
google_key = os.getenv('GOOGLE_KEY')


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


'''
Create categories and make API calls over all resorts
'''

# categories with API specific parameters
categories = {'Restaurants': ['restaurant'],
              'Bars': ['bar'],
              'Spas': ['spa'],
              'Shopping': ['shopping_mall', 'clothing_store', 'department_store'],
              'Medical': ['hospital', 'doctor'],
              'Grocery': ['supermarket', 'liquor_store', 'convenience_store', 'drugstore'],
              'Lodging': ['lodging']}

# import applicable resort dataframe
relative_path = '../../data/'
resort_coordinates = import_data(relative_path, 'resorts_with_coordinates')

'''
# BLOCK COMMENTED OUT AFTER RUNNING TO REDUCE CALLS
# run google function
google_df = source_google_functions.fetch_google_places_data(resort_df, categories, google_key)

# save the information
google_df.to_csv('../../data/google_places.csv', index=False)
'''
