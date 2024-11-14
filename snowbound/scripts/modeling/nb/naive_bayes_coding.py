'''
NAIVE BAYES SCRIPTING
'''

# import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# import sklearn modules
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# import data
resorts = pd.read_csv('../../data/resort_cleaned.csv')
weather = pd.read_csv('../../data/weather_cleaned.csv')
google = pd.read_csv('../../data/google_cleaned.csv')

# function for modeling different scenarios with naive bayes functions
def model_scenario_nb(df_prepared, label, nb_model, plot_title, save_path=None):
    # instantiate selected model
    if nb_model == 'multinomial':
        model = MultinomialNB()
    elif nb_model == 'gaussian':
        model = GaussianNB()
    elif nb_model == 'bernoulli':
        model = BernoulliNB()
    
    # prepare test and train data
    df = df_prepared.copy()
    X = df[[col for col in df.columns if col != label]]
    y = df[label]
    
    # test/train split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # return for splits
    train_return, test_return =  train_test_split(df_prepared, test_size=0.3, random_state=42)
    
    # investigate balance
    label_order = list(df[label].unique())
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    # dataset label balance
    sns.countplot(df, y=label, ax=axes[0], order=label_order)
    axes[0].set_title(f'Label Balance Full - {plot_title}')
    # train label balance
    sns.countplot(y_train, ax=axes[1], order=label_order)
    axes[1].set_title(f'Label Balance Train - {plot_title}')
    # test label balance
    sns.countplot(y_test, ax=axes[2], order=label_order)
    axes[2].set_title(f'Label Balance Test - {plot_title}')
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(f'{save_path}/balance_{plot_title}', dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # train model
    model.fit(X_train, y_train)
    
    # predictions
    predictions = model.predict(X_test)
    
    # probability predictions
    predictions_prob = model.predict_proba(X_test)
    predictions_prob = pd.DataFrame(predictions_prob, columns=model.classes_)
    
    # accuracies
    accuracy = accuracy_score(y_test, predictions)
    
    # confusion matrix
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    plt.figure(figsize=(16, 12))
    disp.plot()
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Accuracy: {accuracy:.2%} for {plot_title}')
    plt.tight_layout()
    
    for text in disp.text_.ravel():
        text.set_fontsize(8)
    
    if save_path is not None:
        plt.savefig(f'{save_path}/cm_{plot_title}', dpi=300, bbox_inches='tight')
        
    plt.show()
    
    return train_return, test_return

'''
DATA PREP Multinomial Naive Bayes
'''

# first ensure date is a datetime object and take the most recent year for each resort
weather['datetime'] = pd.to_datetime(weather['datetime'])
weather['year'] = weather['datetime'].dt.year

# get sum of weather occurences for the most recent year at each resort
weather_types = ['type_freezingrain', 'type_ice', 'type_none', 'type_rain', 'type_snow']
recent_weather_counts = weather.groupby(['year', 'resort'])[weather_types].sum().reset_index()

# take 2023 as the most recent weather year
recent_weather_counts = recent_weather_counts[recent_weather_counts['year']==2023]
recent_weather_counts.reset_index(inplace=True, drop=True)

# merge with resorts to get the label and applicable counts
resort_multinomial = ['Resort', 'Trails Easy', 'Trails Intermediate', 'Trails Difficult', 'Lifts', 'Pass']
recent_weather_merged = pd.merge(recent_weather_counts, resorts[resort_multinomial], left_on='resort', right_on='Resort')

# round the trails and lifts to account for half different types of trails and lifts in the rating system
recent_weather_merged[['Trails Easy', 'Trails Intermediate', 'Trails Difficult', 'Lifts']] = recent_weather_merged[['Trails Easy', 'Trails Intermediate', 'Trails Difficult', 'Lifts']].astype('int')

# drop both the resort columns (one was duplicated during the merging process) and drop year
recent_weather_merged.drop(columns=['Resort', 'resort', 'year'], inplace=True)

# create coy for multinomial
df_multinomial = recent_weather_merged.copy()

# save snippet for gaussian data
df_multinomial.head(10).to_csv('results/multinomial_snippet.csv', index=False)

'''
DATA PREP Gaussian Naive Bayes
'''
# gaussian columns
gaussian_cols = ['temp', 'dew', 'humidity', 'pressure', 'cloudcover', 'icon']

# current icon: 'snow', 'rain', 'clear-day', 'partly-cloudy-day', 'cloudy', 'wind', 'fog'

# change to snow, rain, clear-day, and other
df_icon = weather.copy()
df_icon['icon'] = df_icon['icon'].replace(['partly-cloudy-day', 'cloudy', 'wind', 'fog'], 'other')

# take columns for dataset
df_gaussian = df_icon[gaussian_cols]

df_gaussian.head(10).to_csv('results/gaussian_snippet.csv', index=False)

'''
DATA PREP Bernoulli Naive Bayes
'''
# google columns
google_cols = ['Call Category', 'Initial Category', 'Secondary Category', 'Tertiary Category']
df_bernoulli = google[google_cols]

# now we need to encode the columns
encode_cols = ['Initial Category', 'Secondary Category', 'Tertiary Category']
df_bernoulli = pd.get_dummies(df_bernoulli, columns=encode_cols)

# save snippet for gaussian data
df_bernoulli.head(10).to_csv('results/bernoulli_snippet.csv', index=False)

'''
RUN SCENARIO Multinomial Naive Bayes
'''
train_nb, test_nb = model_scenario_nb(df_multinomial, label='Pass', nb_model='multinomial', plot_title='Multinomial', save_path='results')
train_nb.head(10).to_csv('results/multinomial_train_snippet.csv', index=True)
test_nb.head(10).to_csv('results/multinomial_test_snippet.csv', index=True)

'''
RUN SCENARIO Gaussian Naive Bayes
'''
train_g, test_g = model_scenario_nb(df_gaussian, label='icon', nb_model='gaussian', plot_title='Gaussian', save_path='results')
train_g.head(10).to_csv('results/gaussian_train_snippet.csv', index=True)
test_g.head(10).to_csv('results/gaussian_test_snippet.csv', index=True)

'''
RUN SCENARIO Bernoulli Naive Bayes
'''
train_b, test_b = model_scenario_nb(df_bernoulli, label='Call Category', nb_model='bernoulli', plot_title='Bernoulli', save_path='results')
train_b.head(10).to_csv('results/bernoulli_train_snippet.csv', index=True)
test_b.head(10).to_csv('results/bernoulli_test_snippet.csv', index=True)
