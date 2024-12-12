'''
Ensemble Testing - Random Forest
'''

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn import tree

# use similar approach to SVM section data set up, but don't use PCA, don't simplify to non-min-max, include coordinate info
weather = pd.read_csv('../../data/weather_cleaned.csv')
resorts = pd.read_csv('../../data/resort_cleaned.csv')

# datetime
weather['datetime'] = pd.to_datetime(weather['datetime'])
weather['day_of_year'] = weather['datetime'].dt.dayofyear

# merge coordinates into weather
df = pd.merge(weather, resorts[['Resort', 'Latitude', 'Longitude']], left_on='resort', right_on='Resort')

# drop additional resort column
df.drop(columns=['Resort'], inplace=True)

# lowercase latitude and longitude
df.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'}, inplace=True)

# use rain, snow, clear-day, and other
df_icon = df.copy()
replace_values = ['partly-cloudy-day', 'cloudy', 'wind', 'fog']
df_icon['icon'] = df_icon['icon'].replace(replace_values, 'other')

# now downsample to smallest size
min_prop_count = df_icon['icon'].value_counts().min()
df_balanced = df_icon.groupby(['icon']).apply(lambda row: row.sample(min_prop_count) if len(row) > min_prop_count else row).reset_index(drop=True)

# numerical columns
numeric_cols = ['day_of_year', 'temp', 'dew', 'humidity', 'pressure', 'latitude', 'longitude']
icon_label = ['icon']

# reduce to this set
df_data = df_balanced[numeric_cols + icon_label]

# ensure index is reset
df_data.reset_index(drop=True, inplace=True)

# data snippets
df_data.head(10).to_csv('data_prepared_ensemble.csv', index=False)
train_return, test_return = train_test_split(df_data, test_size=0.3, random_state=42)
train_return.head(10).to_csv('data_train_ensemble.csv', index=False)
test_return.head(10).to_csv('data_test_ensemble.csv', index=False)

'''
Now, perform RF Classification
'''
# X and y
X = df_data[numeric_cols]
y = df_data['icon']

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# random forest model
model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# accuracy
accuracy = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)

# pretty cm
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
plt.figure(figsize=(16, 12))
disp.plot()
plt.xticks(rotation=45, ha='right')
plt.title(f'Accuracy: {accuracy:.2%}')
plt.tight_layout()
plt.savefig('images/shallow_tree_example.png', dpi=300, bbox_inches='tight')
plt.show()

# plot 1
plt.figure(figsize=(40, 25))
tree_model = model.estimators_[0]
tree_plot = tree.plot_tree(tree_model,
                           feature_names=model.feature_names_in_,
                           class_names=model.classes_,
                           filled=True,
                           fontsize=12)
plt.savefig('images/shallow_tree_example_tree_1.png', dpi=500, bbox_inches='tight')
plt.show()

# plot 2
plt.figure(figsize=(40, 25))
tree_model = model.estimators_[1]
tree_plot = tree.plot_tree(tree_model,
                           feature_names=model.feature_names_in_,
                           class_names=model.classes_,
                           filled=True,
                           fontsize=12)
plt.savefig('images/shallow_tree_example_tree_2.png', dpi=500, bbox_inches='tight')
plt.show()

# plot 3
plt.figure(figsize=(40, 25))
tree_model = model.estimators_[2]
tree_plot = tree.plot_tree(tree_model,
                           feature_names=model.feature_names_in_,
                           class_names=model.classes_,
                           filled=True,
                           fontsize=12)
plt.savefig('images/shallow_tree_example_tree_3.png', dpi=500, bbox_inches='tight')
plt.show()

'''
Note that these trees have different base nodes, which illustrates the different subsets either through sampling or feature randomness.
This is difficult to achieve with singular decision trees without dropping complete features.

However, note that this illustrative example has nowhere near pure leaf nodes at the end of the trees.

For visualization purposes, the depth was purposefully kept very shallow for a dataset of this size.

This suggests that training a RFC with a greater max depth could increase the accuracy of the model. However, caution must be used 
to prevent overfitting.
'''

# potential optimal model
model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# accuracy
accuracy = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)

# pretty cm
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
plt.figure(figsize=(16, 12))
disp.plot()
plt.xticks(rotation=45, ha='right')
plt.title(f'Accuracy: {accuracy:.2%}')
plt.tight_layout()
plt.savefig('images/deep_tree_cm.png', dpi=300, bbox_inches='tight')
plt.show()

# look at leaf purities in the models

leaf_df = pd.DataFrame(columns=['Leaf Node', 'Purity', 'Class Distribution', 'Tree Number'])
for tree_num in range(3):
    # get tree in forest
    tree_model = model.estimators_[tree_num].tree_
    
    # leaf nodes (final nodes)
    leaf_indices = np.where(tree_model.children_left == -1)[0]
    
    # leaf purities
    leaf_purities = []
    for leaf in leaf_indices:
        # coutns of each class at the leaf node
        class_counts = tree_model.value[leaf][0]
        # calculate purity as the proportion of the majority class
        purity = class_counts.max() / class_counts.sum()
        leaf_purities.append((leaf, purity, class_counts))
    
    # dataframe to display the leaf node information
    tree_df = pd.DataFrame(leaf_purities, columns=['Leaf Node', 'Purity', 'Class Distribution'])
    tree_df['Tree Number'] = tree_num
    
    # concat into leaf_df
    leaf_df = pd.concat([leaf_df, tree_df], ignore_index=True)

# what's the spread of purities? Examine by the first 3 trees!
plt.figure(figsize=(12, 8))
sns.kdeplot(leaf_df, x='Purity', hue='Tree Number')
plt.title('Deep Tree Purity')
plt.savefig('images/deep_tree_purity.png')
