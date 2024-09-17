# source/source_google_functions

'''
Module for functions using the Google API.

Specific Google API Services:
    - Geocoding API
    - Places API
'''
# import libraries
from dotenv import load_dotenv
import os
import requests
import pandas as pd
import time


# get google api key
env_path = '../config/.env'
load_dotenv(env_path)
google_key = os.getenv('GOOGLE_KEY')


# function to get geocode data
def get_geocode(address, google_key):
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": address,
        "key": google_key
    }
    response = requests.get(base_url, params=params)
    return response.json()


# function to extend get_geocode to parse through list of addresses
def get_coordinates(addresses, google_key):
    # create storage for data results
    results = []
    
    # parse through addresses
    for address in addresses:
        geocode_data = get_geocode(address, google_key)
        if geocode_data['status'] == 'OK':
            location = geocode_data['results'][0]['geometry']['location']
            results.append({
                'Address': address,
                'Latitude': location['lat'],
                'Longitude': location['lng']
            })
        else:
            results.append({
                'Address': address,
                'Latitude': None,
                'Longitude': None
            })
    
    df = pd.DataFrame(results)
    
    return df


# function to get proper address from coordinates
def get_proper_addresses(resort_df, google_key):
    df = resort_df.copy()
    base_url = 'https://maps.googleapis.com/maps/api/geocode/json?'
    address_results = {'Resort': [], 'Proper Address': [], 'Address Components': []}
    for col, row in df.iterrows():
        # get coordinates
        latitude = row['Latitude']
        longitude = row['Longitude']
        resort = row['Resort']
        
        # final url
        url = f'{base_url}latlng={latitude},{longitude}&key={google_key}'
        
        # request
        response = requests.get(url)
        data = response.json()
        
        # extract address
        if 'results' in data and len(data['results']) > 0:
            proper_address = data['results'][0]['formatted_address']
            address_components = data['results'][0]['address_components']
        else:
            proper_address = None
            address_components = None
            
        # add results to dictionary
        address_results['Resort'].append(resort)
        address_results['Proper Address'].append(proper_address)
        address_results['Address Components'].append(address_components)
        
    return pd.DataFrame(address_results)


'''
# running the function
resort_df = pd.read_csv('data/resorts_with_coordinates.csv')
addresses_df = get_proper_addresses(resort_df, google_key)
addresses_df.to_csv('../../data/resorts_proper_addresses.csv', index=False)
'''


# function to get an individual proper address
def get_proper_address(resort, latitude, longitude, google_key):
    # set up
    base_url = 'https://maps.googleapis.com/maps/api/geocode/json?'
    address_results = {'Resort': [], 'Proper Address': [], 'Address Components': []}
    url = f'{base_url}latlng={latitude},{longitude}&key={google_key}'
        
    # request
    response = requests.get(url)
    data = response.json()
    
    # extract address
    if 'results' in data and len(data['results']) > 0:
        proper_address = data['results'][0]['formatted_address']
        address_components = data['results'][0]['address_components']
    else:
        proper_address = None
        address_components = None
        
    # add results to dictionary
    address_results['Resort'].append(resort)
    address_results['Proper Address'].append(proper_address)
    address_results['Address Components'].append(address_components)
        
    return pd.DataFrame(address_results)


# function to stepwise find closest location with proper results
def find_closest_location(resort, latitude, longitude, google_key, step=0.1, max_attempts=10):
    for attempt in range(max_attempts):
        df = pd.DataFrame({'Resort': [resort], 'Latitude': [int(latitude)], 'Longitude': [int(longitude)]})
        address_info = get_proper_addresses(df, google_key)
        if 'address_components' in address_info:
            address_components = address_info['address_components']
            location_info = {}
            for component in address_components:
                if 'country' in component['types']:
                    location_info['country'] = component['long_name']
                if 'administrative_area_level_1' in component['types']:
                    location_info['state'] = component['long_name']
                if 'locality' in component['types']:
                    location_info['city'] = component['long_name']
            
            if location_info and ('state' in location_info or 'city' in location_info):
                return location_info
        
        # adjust coordinates if not
        latitude += step
        longitude += step
        
    return None


# function to return information on google places
def fetch_google_places_data(resort_df, categories, google_key, radius_miles=5, max_requests=150000):
    radius_meters = int(radius_miles * 1609.34)
    results = []
    request_count = 0
    
    for col, row in resort_df.iterrows():
        latitude = row['Latitude']
        longitude = row['Longitude']
        resort = row['Resort']
        for category in categories:
            for place_type in categories[category]:
                url = f'https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={latitude},{longitude}&radius={radius_meters}&type={place_type}&key={google_key}'
                response = requests.get(url)
                request_count += 1
                
                if response.status_code == 200:
                    places = response.json().get('results', [])
                    for place in places:
                        place['resort'] = resort
                        place['category'] = category
                        results.append(place)
                else:
                    print(f"Error: {response.status_code}")
                
                # Check if the request limit is reached
                if request_count >= max_requests:
                    print("Request limit reached. Returning results.")
                    return pd.DataFrame(results)
                
                # Wait a second before the next call
                time.sleep(1)
        
        # progress report
        resort_index = resort_df[resort_df['Resort']==resort].index[0]
        completion = ((resort_index + 1) / resort_df.shape[0]) * 100
        print(f'{completion:.2f}% Complete')
    
    return pd.DataFrame(results)

'''
# running the function
categories = {'Restaurants': ['restaurant'],
              'Bars': ['bar'],
              'Spas': ['spa'],
              'Shopping': ['shopping_mall', 'clothing_store', 'department_store'],
              'Medical': ['hospital', 'doctor'],
              'Grocery': ['supermarket', 'liquor_store', 'convenience_store', 'drugstore'],
              'Lodging': ['lodging']}

# run the function
resort_df = pd.read_csv('data/resorts_with_coordinates.csv')
google_df = fetch_google_places_data(resort_df, categories, google_key)

# save the information
google_df.to_csv('../../google_places.csv', index=False)
'''
