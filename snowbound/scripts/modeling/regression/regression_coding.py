'''
REGRESSION SCRIPTING
'''
# import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# import sklearn modules
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# import data
resorts = pd.read_csv('../../data/resort_cleaned.csv')

'''
IDEA: predict either USA or Canada given ski resort characteristics.
'''

# data columns
data_cols = ['Trails Easy', 'Trails Intermediate', 'Trails Difficult', 'Lifts']
label = 'Country'

resorts_subset = resorts[['Country', 'Trails Easy', 'Trails Intermediate', 'Trails Difficult', 'Lifts']]

# round the trails and lifts to account for half different types of trails and lifts in the rating system
resorts_subset[['Trails Easy', 'Trails Intermediate', 'Trails Difficult', 'Lifts']] = resorts_subset[['Trails Easy', 'Trails Intermediate', 'Trails Difficult', 'Lifts']].astype('int')

# resorts snippet
resorts_subset.head(10).to_csv('results/regression_snippet.csv', index=False)

# split train/test
X = resorts_subset[data_cols]
y = resorts_subset[label]

# perform split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# save split
train_return, test_return = train_test_split(resorts_subset, test_size=0.3, random_state=42)
train_return.head(10).to_csv('results/train_snippet.csv', index=True)
test_return.head(10).to_csv('results/test_snippet.csv', index=True)

# logistc regression
model_log = LogisticRegression()
model_log.fit(X_train, y_train)
predictions_log = model_log.predict(X_test)
predictions_prob_log = model_log.predict_proba(X_test)
accuracy_log = accuracy_score(y_test, predictions_log)
print(accuracy_log)
cm_log = confusion_matrix(y_test, predictions_log)

disp = ConfusionMatrixDisplay(confusion_matrix=cm_log, display_labels=model_log.classes_)
plt.figure(figsize=(16, 12))
disp.plot()
plt.xticks(rotation=45, ha='right')
plt.title(f'Accuracy: {accuracy_log:.2%} for Logistic Regression')
plt.tight_layout()
plt.savefig('results/cm_log', dpi=300, bbox_inches='tight')
plt.show()


# multinomial regression
model_nb = MultinomialNB()
model_nb.fit(X_train, y_train)
predictions_nb = model_nb.predict(X_test)
predictions_prob_nb = model_nb.predict_proba(X_test)
accuracy_nb = accuracy_score(y_test, predictions_nb)
print(accuracy_nb)
cm_nb= confusion_matrix(y_test, predictions_nb)

disp = ConfusionMatrixDisplay(confusion_matrix=cm_nb, display_labels=model_nb.classes_)
plt.figure(figsize=(16, 12))
disp.plot()
plt.xticks(rotation=45, ha='right')
plt.title(f'Accuracy: {accuracy_nb:.2%} for Multinomial Naive Bayes')
plt.tight_layout()
plt.savefig('results/cm_nb', dpi=300, bbox_inches='tight')
plt.show()
