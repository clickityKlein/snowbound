'''
DECISION TREES SCRIPTING
'''

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import sklearn modules
from sklearn.tree import DecisionTreeClassifier
import pydotplus
import graphviz
from dtreeviz.models.shadow_decision_tree import ShadowDecTree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn.model_selection import GridSearchCV


# function for modeling different scenarios with decision trees
def model_scenario_dt(df_prepared, label, criterion='gini', splitter='best', class_weight=None, plot_title='Default', save_path='results'):
    # prepare test and train data
    df = df_prepared.copy()
    X = df[[col for col in df.columns if col != label]]
    y = df[label]
    
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # return for splits
    train_return, test_return = train_test_split(df_prepared, test_size=0.3, random_state=42)
    
    model = DecisionTreeClassifier(criterion=criterion, splitter=splitter, class_weight=class_weight)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    prediction_prob = model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, prediction)
    cm = confusion_matrix(y_test, prediction)
    
    # accuracy - confusion matrix plots
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    plt.figure(figsize=(16, 12))
    disp.plot()
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Accuracy: {accuracy:.2%} for {plot_title}')
    plt.tight_layout()
    plt.savefig(f'{save_path}/cm_{plot_title}', dpi=300, bbox_inches='tight')
    plt.show()
    
    # tree plot
    plt.figure(figsize=(30, 20))
    # ignore warning for never used on tree_plot, it is used for saving
    tree_plot = tree.plot_tree(model,
                               feature_names=X_train.columns.values,
                               class_names=model.classes_,
                               filled=True)
    plt.title(f'{plot_title} - Accuracy: {accuracy:.2%}')
    plt.savefig(f'{save_path}/{plot_title}.png', dpi=500, bbox_inches='tight')
    plt.show()
    
    # return train and test results
    return train_return, test_return


# import data
resorts = pd.read_csv('../../data/resort_cleaned.csv')
weather = pd.read_csv('../../data/weather_cleaned.csv')
weather['datetime'] = pd.to_datetime(weather['datetime'])
weather['month'] = weather['datetime'].dt.month

# getting the data ready
# average weather variable by month and resort
weather_vars = ['temp', 'dew', 'humidity', 'precip', 'snow', 'snowdepth', 'windspeed', 'pressure', 'cloudcover', 'visibility']
weather_avg = weather.groupby(['month', 'resort'])[weather_vars].mean().reset_index()

# merge in pass type
weather_avg_merged = pd.merge(weather_avg, resorts[['Resort', 'Pass', 'Region', 'Elevation Low', 'Elevation High']], left_on='resort', right_on='Resort')

# drop resorts
weather_avg_merged.drop(columns=['resort', 'Resort'], inplace=True)

# encode region
weather_resorts_encoded = pd.get_dummies(weather_avg_merged, columns=['Region'])

# investigate balance
sns.countplot(weather_resorts_encoded, y='Pass')

# drop Ikon and Epic -> we'll create a good model to identify possible Independent Resorts
weather_resorts = weather_resorts_encoded[weather_resorts_encoded['Pass'] != 'Other']
weather_resorts.reset_index(drop=True, inplace=True)

# investigate balance
plt.figure(figsize=(10, 8))
sns.countplot(weather_resorts, y='Pass', stat='proportion')
plt.title('Label Balance - Full Dataset')
plt.xlabel('Counts')
plt.savefig('results/label_balance_full.png', dpi=300)
plt.show()

# variables to model with
model_vars = ['month', 'temp', 'dew', 'humidity', 'precip', 'snow', 'snowdepth', 'windspeed', 'pressure', 'cloudcover', 'visibility', 'Elevation Low', 'Elevation High']
model_vars = model_vars + [col for col in weather_resorts.columns if col.startswith('Region')]
label = 'Pass'

# df_prepared
df_prepared = weather_resorts[model_vars + [label]]
df_prepared.head(10).to_csv('results/dt_prepared_snippet.csv', index=False)

'''
Scenario Default
'''
train_default, test_default = model_scenario_dt(df_prepared, label, criterion='gini', splitter='best', class_weight=None, plot_title='Default')
train_default.head(10).to_csv('results/dt_default_train_snippet.csv', index=True)
test_default.head(10).to_csv('results/dt_default_test_snippet.csv', index=True)

'''
Scenario Entropy
'''
train_entropy, test_entropy = model_scenario_dt(df_prepared, label, criterion='entropy', splitter='best', class_weight=None, plot_title='Entropy')

'''
Scenario Log Loss
'''
train_log_loss, test_log_loss = model_scenario_dt(df_prepared, label, criterion='log_loss', splitter='best', class_weight=None, plot_title='Log Loss')

'''
Scenario Log Loss & Balanced
'''
train_log_balance, test_log_balance = model_scenario_dt(df_prepared, label, criterion='log_loss', splitter='best', class_weight='balanced', plot_title='Log Loss with Balance')

'''
Scenario Log Loss & Balanced & Random Splitter
'''
train_log_balance_random, test_log_balance_random = model_scenario_dt(df_prepared, label, criterion='log_loss', splitter='random', class_weight='balanced', plot_title='Log Loss with Balance with Random')

'''
Best Scenario
'''
# prepare test and train data
df = df_prepared.copy()
X = df[[col for col in df.columns if col != label]]
y = df[label]

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# parameters to test
parameters = {'criterion': ('gini', 'entropy', 'log_loss'),
              'splitter': ('best', 'random'),
              'class_weight': (None, 'balanced')}

# instantiate the hyper model
model_hyper = GridSearchCV(DecisionTreeClassifier(), parameters)

# train the hyper model
model_hyper.fit(X_train, y_train)

# results of hyper model
model_hyper_results = pd.DataFrame(model_hyper.cv_results_)

# model best paramaters
print(model_hyper.best_params_)

'''
{'class_weight': 'balanced', 'criterion': 'entropy', 'splitter': 'best'}
'''

# best case scenario training
train_best, test_best = model_scenario_dt(df_prepared, label, criterion='entropy', splitter='best', class_weight='balanced', plot_title='Best Case')

'''
Predictions with the Best Case
'''
# create model
model = DecisionTreeClassifier(criterion='entropy', class_weight='balanced')
model.fit(X_train, y_train)

# merge data for the suitable dataframe
weather_suitable = pd.merge(weather_avg, resorts[['Resort', 'Pass', 'Region', 'Elevation Low', 'Elevation High']], left_on='resort', right_on='Resort')

# encode region
weather_suitable_encoded = pd.get_dummies(weather_suitable, columns=['Region'])

# create copy
weather_resorts_suitable = weather_suitable_encoded.copy()

# subset to Other
weather_resorts_suitable = weather_resorts_suitable[weather_resorts_suitable['Pass'] == 'Other']
weather_resorts_suitable.reset_index(drop=True, inplace=True)

# predict to find the suitable
prediction = model.predict(weather_resorts_suitable[model_vars])

# spread of predicted pass per resort
predicted_passes = weather_resorts_suitable[['month', 'Resort']]
predicted_passes['Predicted'] = prediction

# for each resort, give percentage breakdown of either pass
percentage = predicted_passes.groupby('Resort')['Predicted'].value_counts(normalize=True).reset_index(name='Percentage')
percentage.to_csv('results/predicted_label_percents.csv', index=False)
max_percentage = percentage.pivot(index='Resort', columns='Predicted', values='Percentage').max(axis=1).reset_index(name='Max_Percent')

plt.figure(figsize=(10, 8))
sns.kdeplot(max_percentage, x='Max_Percent', fill=True)
plt.xlabel('Percent')
plt.title('Resort Choice by Percent')
plt.savefig('results/percent_choice.png', dpi=300)

'''
OVERVIEW SMALL SAMPLE
'''

# sample data
sample_data = {'age': [25, 34, 45, 52, 23],
               'blood_pressure': [120, 140, 130, 150, 115],
               'cholesterol': [180, 220, 200, 240, 170],
               'disease': ['False', 'True', 'False', 'True', 'False']}

df_sample = pd.DataFrame(sample_data)

# scenario
train_sample, test_sample = model_scenario_dt(df_sample, label='disease', criterion='entropy', splitter='best', class_weight=None, plot_title='Entropy Sample')
