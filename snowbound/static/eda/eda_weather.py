# eda/eda_weather

'''
Module for exploratory data analysis with the weather data.
'''

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

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

# weather
weather = import_data(relative_path, 'weather_cleaned')
# make sure datetime has proper datetime format
weather['datetime'] = pd.to_datetime(weather['datetime'])

# resorts
resorts = import_data(relative_path, 'resort_cleaned')
resorts_trimed = [resort for resort in resorts['Resort'].unique() if resort in weather['resort'].unique()]
weather_trimmed = weather[weather['resort'].isin(resorts_trimed)]

# merge resort to get location columns
weather_merged = pd.merge(weather_trimmed,
                          resorts[['Resort', 'Country', 'state_province_territory', 'Region', 'Pass']],
                          left_on='resort', right_on='Resort', how='left')

test = weather_merged.iloc[:5]

'''
check for outliers on numeric columns:    
    - tempmax
    - tempmin
    - temp
    - feelslikemax
    - feelslikemin
    - feelslike
    - dew
    - humidity
    - precip
    - precipcover
    - snow
    - snowdepth
    - windgust
    - windspeed
    - winddir
    - pressure
    - cloudcover
    - visibility
    - moonphase
'''

base_color = sns.color_palette()[0]

# numeric spread
numeric_cols = ['tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike', 'dew', 'humidity', 'precip',
                'precipcover', 'snow', 'snowdepth', 'windgust', 'windspeed', 'winddir', 'pressure', 'cloudcover', 'visibility',
                'solarradiation', 'solarenergy', 'uvindex', 'moonphase']
weather_melted = pd.melt(weather[numeric_cols])
plt.figure(figsize=(12, 8))
sns.boxplot(x='value', y='variable', data=weather_melted, color=base_color)
plt.xlabel('Observation')
plt.ylabel('Column')
plt.title('Weather Numeric Variable Spread')
plt.savefig('weather_numeric_spread', dpi=300)
plt.show()

# date additions
weather_merged['month'] = weather_merged['datetime'].dt.month
weather_merged['year'] = weather_merged['datetime'].dt.year
weather_merged['day'] = weather_merged['datetime'].dt.dayofyear

# average tempmax and tempmin by year (DECENT)
temp_year = weather_merged.groupby(['year'])[['tempmin', 'tempmax']].mean().reset_index()
temp_year_melt = pd.melt(temp_year, id_vars='year', value_vars=['tempmin', 'tempmax'])
temp_year_melt['variable'] = temp_year_melt['variable'].replace({'tempmin': 'Minimum Temperature', 'tempmax': 'Maximum Temperature'})
plt.figure(figsize=(12, 8))
sns.lineplot(data=temp_year_melt, x='year', y='value', hue='variable')
plt.xlabel('Year')
plt.ylabel('Temperature (F)')
plt.legend(title='Temperature Type')
plt.title('Average Minimum and Maximum Temperatures Over Years')
plt.savefig("average_temps_over_years.png", dpi=300)
plt.show()

# average snow by region (DECENT)
snow_region = weather_merged.groupby(['Region', 'Country'])[['snow']].mean().reset_index()
plt.figure(figsize=(12, 8))
sns.barplot(data=snow_region.sort_values(['Country', 'snow'], ascending=False), x='snow', y='Region', hue='Country')
plt.xlabel('Region')
plt.ylabel('Snow (in)')
plt.title('Average Snow by Region')
plt.savefig('average_snow_by_region.png', dpi=300)
plt.show()

# average snow by pass (DECENT)
snow_pass = weather_merged.groupby(['Pass'])[['snow']].mean().reset_index()
plt.figure(figsize=(12, 8))
sns.barplot(data=snow_pass.sort_values(['snow'], ascending=False), x='snow', y='Pass', color=base_color)
plt.xlabel('Snow (in)')
plt.ylabel('Pass')
plt.title('Average Snow by Pass Type')
plt.savefig('average_snow_by_pass.png', dpi=300)
plt.show()

'''
# temperature to snow (use average by day, hue by humidity)
snow_day = weather_merged.groupby(['day'])[['snow', 'temp', 'humidity']].mean().reset_index()
snow_day['humidity_bins'] = pd.cut(snow_day['humidity'], bins=3, labels=['Low', 'Medium', 'High'])
sns.scatterplot(data=snow_day, x='temp', y='snow', hue='humidity_bins')

# temperature to snow with humidity hue (DECENT)
snow_temp = weather_merged.copy()
snow_temp['humidity_bins'] = pd.cut(snow_temp['humidity'], bins=3, labels=['Low', 'Medium', 'High'])
custom_palette = {'Low': 'grey', 'Medium': 'lightblue', 'High': 'darkblue'}
sns.scatterplot(data=snow_temp, x='temp', y='snow', hue='humidity_bins', palette=custom_palette)
'''

# snow_temp
snow_temp = weather_merged.copy()
custom_palette = {'Low': 'grey', 'Medium': 'lightblue', 'High': 'darkblue'}

# temperature to snow with humidity hue - facets - (REAL DECENT)
g = sns.FacetGrid(snow_temp, col='humidity_bins', row='Region', palette=custom_palette, height=4, aspect=1.2)
g.map(sns.scatterplot, 'temp', 'snow',).add_legend()
g.set_titles(col_template="{col_name} Humidity", row_template="{row_name} Region")
g.fig.suptitle('Temperature vs Snowfall by Humidity Levels and Regions', y=1.05)
g.set_axis_labels("Temperature (F)", "Snow (in)")
g.savefig("temperature_vs_snowfall_by_humidity_and_region.png", dpi=300)
plt.show()

# Convert 'Country' to a categorical type
snow_temp['Country'] = snow_temp['Country'].astype('category')

# Create the FacetGrid with hue for 'Country'
g = sns.FacetGrid(snow_temp, col='humidity_bins', row='Region', height=4, aspect=1.2, hue='Country', hue_order=['United States', 'Canada'])
g.map(sns.scatterplot, 'temp', 'snow').add_legend()
g.set_titles(col_template="{col_name} Humidity", row_template="{row_name} Region")
g.fig.suptitle('Temperature vs Snowfall by Humidity Levels and Regions', y=1.05)
g.set_axis_labels("Temperature (F)", "Snow (in)")
g.savefig("temperature_vs_snowfall_by_humidity_and_region_and_country.png", dpi=300)
plt.show()
