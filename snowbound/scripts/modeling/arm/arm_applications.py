'''
ARM APPLICATIONS
'''

## IMPORTS ##

# libraries
import numpy as np
import pandas as pd
import sys
import os

# custom function library
from arm_functions import *

# data
# relative path to data
relative_path = '../../data'

# resorts
resorts_path = os.path.join(relative_path, 'resort_cleaned.csv')
resorts = pd.read_csv(resorts_path)

# google places - types
google_unpacked_path = os.path.join(relative_path, 'google_unpacked.csv')
google_unpacked = pd.read_csv(google_unpacked_path)
google_unpacked.head(10).to_csv('results/google_unpacked_initial.csv', index=False)

# google unpacked needs an additional step from the cleaning process before
google_unpacked.drop_duplicates(subset=['latitude', 'longitude', 'name'], inplace=True)
google_unpacked.reset_index(drop=True, inplace=True)

# turn types into literal lists
google_unpacked['types'] = google_unpacked['types'].apply(lambda row: ast.literal_eval(row))

# pair with google top three categories
google_unpacked[['Initial Category', 'Secondary Category', 'Tertiary Category']] = google_unpacked['types'].apply(lambda row: row[:3]).apply(pd.Series)


# google places - cleaned
google_path = os.path.join(relative_path, 'google_cleaned.csv')
google = pd.read_csv(google_path)

# match types into main google
google_unpacked_match = google_unpacked[['latitude', 'longitude', 'types', 'call_category', 'Initial Category', 'Secondary Category', 'Tertiary Category']]
google_unpacked_match.rename(columns={'latitude': 'Latitude',
                                      'longitude': 'Longitude',
                                      'call_category': 'Call Category'},
                             inplace=True)

google_merged = google_unpacked_match.merge(google, on=['Latitude', 'Longitude', 'Call Category', 'Initial Category', 'Secondary Category', 'Tertiary Category'])
google_merged.drop_duplicates(subset=['Resort', 'Latitude', 'Longitude', 'Name'], inplace=True)
drop_columns = ['Initial Category', 'Secondary Category', 'Tertiary Category', 'Latitude', 'Longitude', 'Name', 'rating', 'total_ratings']
google_merged.drop(columns=drop_columns, inplace=True)

# merge in more potential labels
google_merged = google_merged.merge(resorts[['Resort', 'Country', 'Pass', 'Region']], on='Resort')
google_merged.to_csv('results/google_merged.csv', index=False)

# save as google_merged
google_merged.head(10).to_csv('results/google_merged.csv', index=False)

'''
RUN FUNCTIONS
'''
## BASE CATEGORIES ##
# initialization
con_base = TransactionEncoder()
con_arr_base = con_base.fit_transform(google_merged['types'])
df_transformed_base = pd.DataFrame(con_arr_base, columns=con_base.columns_)

# run apriori
apriori_base = ml_apriori(df_transformed_base, use_colnames=True, min_support=0.01)

# run association rules
rules_base = ml_association(apriori_base, metric='confidence', min_threshold=0.01)

# save rules base
rules_base.to_csv('results/rules_base.csv', index=False)

# top support
top_15_support = rules_base.sort_values(by='support', ascending=False).head(15)
top_15_support_lifted = rules_base[rules_base['lift']>1].sort_values(by='support', ascending=False).head(15)

# top confidence
top_15_confidence = rules_base.sort_values(by='confidence', ascending=False).head(15)
top_15_confidence_lifted = rules_base[rules_base['lift']>1].sort_values(by='confidence', ascending=False).head(15)

# top lift
top_15_lift = rules_base.sort_values(by='lift', ascending=False).head(15)

# visualize top 15s with lift greater than 1
create_network_pyvis(top_15_support_lifted, save_path='results/top_15_support_lifted.html')
create_network_pyvis(top_15_confidence_lifted, save_path='results/top_15_confidence_lifted.html')
create_network_pyvis(top_15_lift, save_path='results/top_15_lift.html')

## ADD LABEL: API CALLS ##
# create rules by function
rules_calls_categories = rules_from_label(google_merged, 'types', 'Call Category', prefix='call_', apriori_min_support=0.01, rules_min_threshold=0.01, rules_min_metric='confidence')

# save rule calls categories
rules_calls_categories.to_csv('results/rules_calls.csv', index=False)

# plot rules by different parameters
# bulk call
G_call = create_network_pyvis(rules_calls_categories, save_path='results/label_call.html', color_palette=True)
# restaurant call
G_restaurants = create_network_pyvis(rules_calls_categories, antecedent_label='call_restaurants', save_path='results/label_call_restaurants.html')
# bar call
G_bars = create_network_pyvis(rules_calls_categories, antecedent_label='call_bars', save_path='results/label_call_bars.html')
# shopping call
G_shopping = create_network_pyvis(rules_calls_categories, antecedent_label='call_shopping', save_path='results/label_call_shopping.html')
# medical call
G_medical = create_network_pyvis(rules_calls_categories, antecedent_label='call_medical', save_path='results/label_call_medical.html')
# grocery call
G_grocery = create_network_pyvis(rules_calls_categories, antecedent_label='call_grocery', save_path='results/label_call_grocery.html')
# lodging call
G_lodging = create_network_pyvis(rules_calls_categories, antecedent_label='call_lodging', save_path='results/label_call_lodging.html')

## ADD LABEL: COUNTRY ##
# create rules by function
rules_country_categories = rules_from_label(google_merged, 'types', 'Country', prefix='', apriori_min_support=0.01, rules_min_threshold=0.01, rules_min_metric='confidence')

# save rule calls categories
rules_country_categories.to_csv('results/rules_country.csv', index=False)

## ADD LABEL: PASS ##
# create rules by function
rules_pass_categories = rules_from_label(google_merged, 'types', 'Pass', prefix='', apriori_min_support=0.01, rules_min_threshold=0.01, rules_min_metric='confidence')

# save rule calls categories
rules_pass_categories.to_csv('results/rules_pass.csv', index=False)

## ADD LABEL: REGION ##
# create rules by function
rules_region_categories = rules_from_label(google_merged, 'types', 'Region', prefix='', apriori_min_support=0.01, rules_min_threshold=0.01, rules_min_metric='confidence')

# save rule calls categories
rules_region_categories.to_csv('results/rules_region.csv', index=False)
