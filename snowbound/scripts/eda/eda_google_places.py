# eda/eda_google_places

'''
Module for exploratory data analysis with the google places data.
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

# google
google = import_data(relative_path, 'google_cleaned')

# resorts
resorts = import_data(relative_path, 'resort_cleaned')

# merge together
google_merged = pd.merge(google, resorts, on='Resort')

'''
Visuals
'''
# base color
base_color = sns.color_palette()[0]

# rating by call category - average scores
category_rating = google_merged.groupby(['Call Category', 'Country', 'Region'])['rating'].mean().reset_index()

# initial bar chart
plt.figure(figsize=(12, 8))
sns.barplot(data=category_rating.sort_values(['Country', 'rating'], ascending=False), x='rating', y='Call Category', hue='Country')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.title('Average Category Ratings by Country')
plt.xlabel('Rating')
plt.savefig("business_ratings_by_country.png", dpi=300, bbox_inches='tight')
plt.show()


# catplot
g = sns.catplot(
    data=category_rating.sort_values(['Country', 'Region', 'rating'], ascending=False), kind="bar",
    x="rating", y="Call Category",
    hue="Country", col="Region",
    col_wrap=5, height=4, aspect=1,
    orient='h', hue_order=['United States', 'Canada'])

# labels and titles
g.set_axis_labels("Rating", "Call Category")
g.set_titles("{col_name}")

# ratings
for ax in g.axes.flat:
    for p in ax.patches:
        ax.text(p.get_width() + 0.1, p.get_y() + p.get_height() / 2,
                f'   {p.get_width():.1f}', ha='center', va='center')
# save
g.savefig("business_ratings_by_region", dpi=300)

# show
plt.show()
