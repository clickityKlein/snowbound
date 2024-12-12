'''
SVM
'''

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


'''
Predictions:
- use weather data to predict weather type
- make another note about the inherently binary classifier, but sklearn's model ensembles svm models together
'''

# data
df = pd.read_csv('../../data/weather_cleaned.csv')

# datetime
df['datetime'] = pd.to_datetime(df['datetime'])
df['day_of_year'] = df['datetime'].dt.dayofyear

# label is icon, look at proportions (balance)
df['icon'].value_counts()

'''
rain                 234099
partly-cloudy-day    233629
clear-day            159724
snow                 137150
cloudy                19445
wind                   4809
fog                     179
'''

# use rain, snow, clear-day, and other
df_icon = df.copy()
replace_values = ['partly-cloudy-day', 'cloudy', 'wind', 'fog']
df_icon['icon'] = df_icon['icon'].replace(replace_values, 'other')

# look at proportions (balance)
df_icon['icon'].value_counts()

'''
other        258062
rain         234099
clear-day    159724
snow         137150
'''

# now downsample to smallest size
min_prop_count = df_icon['icon'].value_counts().min()
df_balanced = df_icon.groupby(['icon']).apply(lambda row: row.sample(min_prop_count) if len(row) > min_prop_count else row).reset_index(drop=True)

# look at proportions (balance)
df_balanced['icon'].value_counts()
'''
clear-day    137150
other        137150
rain         137150
snow         137150
'''

# still a decent amount of data! 548600 rows
df_balanced.shape[0]

# now we'll reduce the columns to numeric columns only and perform pca
# don't use columns directly related to the icon label though!
# this includes: precip, prcipprob, precipcover, snow, snowdepth, windgust, windspeed, cloudcover, visibility
numeric_cols = ['tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike', 'dew', 'humidity', 'pressure', 'day_of_year']
icon_label = ['icon']

# pca
scaler = StandardScaler()
df_normal = scaler.fit_transform(df_balanced[numeric_cols])
pca = PCA()
pca_projection = pca.fit_transform(df_normal)
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)
df_explained = pd.DataFrame({
                'principal_components': [f'principal_component_{col+1}' for col in range(len(explained_variance))],
                'explained_variance': explained_variance,
                'cumulative_variance': cumulative_variance
                })

# save df_explained
df_explained.to_csv('explained_variance_df.csv', index=False)

# plot
plt.figure(figsize=(12, 8))
sns.barplot(data=df_explained, y='explained_variance', x='principal_components', color=sns.color_palette()[0], label='Explained Variance')
sns.lineplot(data=df_explained, y='cumulative_variance', x='principal_components', color='red', label='Cumulative Variance')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance')
plt.xticks(range(df_explained.shape[0]), range(1, df_explained.shape[0] + 1))
plt.legend(loc=True)
plt.savefig('images/explained_variance.png', dpi=300, bbox_inches='tight')
plt.show()

# create results dataframe with the label as icon
df_pca = pd.DataFrame(pca_projection)
df_pca_cols = [f'principal_component_{component}' for component in range(1, df_pca.shape[1] + 1)]
df_pca.columns = df_pca_cols
df_results = df_pca[['principal_component_1', 'principal_component_2', 'principal_component_3']]
df_results = pd.concat([df_results, df_balanced['icon']], axis=1)

# report svm data
df_results.head(10).to_csv('data_prepared_svm.csv', index=False)
train_return, test_return = train_test_split(df_results, test_size=0.3, random_state=42)
train_return.head(10).to_csv('data_train_svm.csv', index=False)
test_return.head(10).to_csv('data_test_svm.csv', index=False)

# subset for ease of plotting
df_results_subset = df_results.sample(frac=0.01, random_state=42)

# visualize 2D results - use a 10% subset
sns.scatterplot(df_results_subset, x='principal_component_1', y='principal_component_2', hue='icon')

# function to report svm results
def report_svm_results(model, X_train, X_test, y_train, y_test, plot_title, save_path=None):
    # predictions and scores
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    
    # pretty confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    plt.figure(figsize=(16, 12))
    disp.plot()
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Accuracy: {accuracy:.2%} for {plot_title}')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    

# train test split
X = df_results_subset[['principal_component_1', 'principal_component_2']]
y = df_results_subset['icon']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# poly model 1
poly_model_1 = SVC(C=1.0, kernel='poly', degree=2)
poly_model_1.fit(X_train, y_train)
report_svm_results(poly_model_1, X_train, X_test, y_train, y_test, 'SVM: Polynomial - Degree: 2 - C: 1.0', save_path='images/poly_2_1.png')

# poly model 2
poly_model_2 = SVC(C=2.0, kernel='poly', degree=2)
poly_model_2.fit(X_train, y_train)
report_svm_results(poly_model_2, X_train, X_test, y_train, y_test, 'SVM: Polynomial - Degree: 2 - C: 2.0', save_path='images/poly_2_2.png')

# poly model 3
poly_model_3 = SVC(C=5.0, kernel='poly', degree=2)
poly_model_3.fit(X_train, y_train)
report_svm_results(poly_model_3, X_train, X_test, y_train, y_test, 'SVM: Polynomial - Degree: 2 - C: 5.0', save_path='images/poly_2_5.png')

# poly model 4
poly_model_4 = SVC(C=2.0, kernel='poly', degree=3)
poly_model_4.fit(X_train, y_train)
report_svm_results(poly_model_4, X_train, X_test, y_train, y_test, 'SVM: Polynomial - Degree: 3 - C: 2.0', save_path='images/poly_3_2.png')

# poly model 5
poly_model_5 = SVC(C=2.0, kernel='poly', degree=4)
poly_model_5.fit(X_train, y_train)
report_svm_results(poly_model_5, X_train, X_test, y_train, y_test, 'SVM: Polynomial - Degree: 4 - C: 2.0', save_path='images/poly_5_2.png')

# rbf model 1
rbf_model_1 = SVC(C=1.0, kernel='rbf')
rbf_model_1.fit(X_train, y_train)
report_svm_results(rbf_model_1, X_train, X_test, y_train, y_test, 'SVM: RBF - C: 1.0', save_path='images/rbf_1.png')

# rbf model 2
rbf_model_2 = SVC(C=2.0, kernel='rbf')
rbf_model_2.fit(X_train, y_train)
report_svm_results(rbf_model_2, X_train, X_test, y_train, y_test, 'SVM: RBF - C: 2.0', save_path='images/rbf_2.png')

# rbf model 3
rbf_model_3 = SVC(C=3.0, kernel='rbf')
rbf_model_3.fit(X_train, y_train)
report_svm_results(rbf_model_3, X_train, X_test, y_train, y_test, 'SVM: RBF - C: 3.0', save_path='images/rbf_3.png')

# sigmoid model 1
rbf_model_1 = SVC(C=1.0, kernel='sigmoid')
rbf_model_1.fit(X_train, y_train)
report_svm_results(rbf_model_1, X_train, X_test, y_train, y_test, 'SVM: Sigmoid - C: 1.0', save_path='images/sigmoid_1.png')

# sigmoid model 2
rbf_model_2 = SVC(C=2.0, kernel='sigmoid')
rbf_model_2.fit(X_train, y_train)
report_svm_results(rbf_model_2, X_train, X_test, y_train, y_test, 'SVM: Sigmoid - C: 2.0', save_path='images/sigmoid_2.png')

# sigmoid model 3
rbf_model_3 = SVC(C=0.5, kernel='sigmoid')
rbf_model_3.fit(X_train, y_train)
report_svm_results(rbf_model_3, X_train, X_test, y_train, y_test, 'SVM: Sigmoid - C: 0.5', save_path='images/sigmoid_05.png')

# sigmoid model 4
rbf_model_4 = SVC(C=0.1, kernel='sigmoid')
rbf_model_4.fit(X_train, y_train)
report_svm_results(rbf_model_4, X_train, X_test, y_train, y_test, 'SVM: Sigmoid - C: 0.1', save_path='images/sigmoid_01.png')

# test on 3 component PCA - rbf default
df_results_large = df_pca[['principal_component_1', 'principal_component_2', 'principal_component_3']]
df_results_large = pd.concat([df_results_large, df_balanced['icon']], axis=1)

# 3 component pca
df_results_subset_large = df_results_large.sample(frac=0.01, random_state=42)
X = df_results_subset_large[['principal_component_1', 'principal_component_2', 'principal_component_3']]
y = df_results_subset_large['icon']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# model
rbf_model = SVC(C=1.0, kernel='rbf')
rbf_model.fit(X_train, y_train)
report_svm_results(rbf_model, X_train, X_test, y_train, y_test, 'SVM: RBF Large Subset', save_path='images/rbf_large.png')

# 3 component pca with larger subset
df_results_subset_large = df_results_large.sample(frac=0.05, random_state=42)
X = df_results_subset_large[['principal_component_1', 'principal_component_2', 'principal_component_3']]
y = df_results_subset_large['icon']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# model
rbf_model = SVC(C=1.0, kernel='rbf')
rbf_model.fit(X_train, y_train)
report_svm_results(rbf_model, X_train, X_test, y_train, y_test, 'SVM: RBF Larger Subset', save_path='images/rbf_larger.png')
