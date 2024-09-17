# eda/eda_ski_resorts

'''
Module for exploratory data analysis with the ski resorts data.
'''

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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

## IMPORT DATA ## 
relative_path = '../../data'

# resorts
resorts = import_data(relative_path, 'resort_cleaned')

# weather
weather = import_data(relative_path, 'weather_cleaned')

# google
google = import_data(relative_path, 'google_cleaned')

'''
EDA
'''
# base color
base_color = sns.color_palette()[0]

# total rating by resort
resorts_melted = pd.melt(resorts, id_vars=['Country', 'Region', 'Pass'], value_vars=['Overall Rating'])
plt.figure(figsize=(12, 8))
sns.boxplot(x='value', y='Country', data=resorts_melted, color=base_color)
plt.xlabel('Resort Rating')
plt.title('Resort Ratings by Country')
plt.savefig('resort_ratings_by_country.png', dpi=300)
plt.show()

# total rating by region and country
plt.figure(figsize=(12, 8))
sns.boxplot(x='value', y='Region', data=resorts_melted.sort_values(['Country', 'value'], ascending=False), hue='Country')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.xlabel('Resort Rating')
plt.title('Resort Ratings by Region')
plt.savefig('resort_ratings_by_region.png', dpi=300, bbox_inches='tight')
plt.show()

# total rating by pass
plt.figure(figsize=(12, 8))
sns.boxplot(x='value', y='Pass', data=resorts_melted, color=base_color)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.xlabel('Resort Rating')
plt.title('Resort Ratings by Pass')
plt.savefig('resort_ratings_by_pass.png', dpi=300, bbox_inches='tight')
plt.show()


# overall rating vs. price
plt.figure(figsize=(12, 8))
sns.scatterplot(data=resorts, x='Overall Rating', y='Price', hue='Country')
plt.title('Rating vs. Price by Country')
plt.savefig('rating_vs_price_by_country.png', dpi=300)
plt.show()

# correlation plot of the ratings
rating_cols = ['Overall Rating', 'Resort Size', 'Run Variety', 'Lifts Quality']
ratings_corr = resorts[rating_cols].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(ratings_corr, annot=True, cmap='coolwarm')
plt.title('Resort Rating Heatmap')
plt.savefig('resort_rating_heatmap.png', dpi=300)
plt.show()

# correlation plot of all numerics
numeric_cols = ['Overall Rating', 'Elevation Difference', 'Elevation Low', 'Elevation High',
                'Trails Total', 'Trails Easy', 'Trails Intermediate', 'Trails Difficult', 'Lifts',
                'Price', 'Resort Size', 'Run Variety', 'Lifts Quality']

numeric_corr = resorts[numeric_cols].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(numeric_corr, cmap='coolwarm')
plt.title('Resort Numerical Variables Heatmap')
plt.savefig('resort_numerical_heatmap.png', dpi=300)
plt.show()

'''
Massive Heatmap for all Numerical Values in Resorts, Google, and Weather
'''
# get average by resort for google ratings - categorical
google_resorts = google.groupby(['Call Category', 'Resort'])['rating'].mean().reset_index()
google_pivot = google_resorts.pivot(index='Resort', values=['rating'], columns=['Call Category'])
google_pivot = google_pivot.reset_index()
google_pivot.columns = ['Resort'] + [f'rating_{col[1].lower()}' for col in google_pivot.columns[1:]]

# get average by resort for weather numeric variables
weather_cols = ['tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike', 'dew', 'humidity', 'precip',
                'precipcover', 'snow', 'snowdepth', 'windgust', 'windspeed', 'winddir', 'pressure', 'cloudcover', 'visibility',
                'solarradiation', 'solarenergy', 'uvindex']

weather_resorts = weather.groupby(['resort'])[weather_cols].mean().reset_index()

# merge together
massive_merge = pd.merge(resorts, google_pivot, on='Resort')
massive_merge = pd.merge(massive_merge, weather_resorts, left_on='Resort', right_on='resort')

# drop non-numeric columns
massive_merge = massive_merge.select_dtypes(include=[np.number])

# correlation
massive_corr = massive_merge.corr()

# heatmap
plt.figure(figsize=(24, 16))
sns.heatmap(massive_corr, cmap='coolwarm')
plt.title('Total Heatmap')
plt.savefig('total_heatmap.png', dpi=300)
plt.show()

# save correlation matrix as well
massive_corr.to_csv('total_correlation.csv', index=False)
