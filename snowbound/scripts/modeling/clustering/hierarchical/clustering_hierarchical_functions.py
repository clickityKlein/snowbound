'''
HIERARCHICAL CLUSTERING FUNCTIONS
'''

'''
Imports
'''

# standard libraries
import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

# clustering libraries
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import chi2_contingency

# custom pca functions
sys.path.insert(0, '../pca')
from pca_functions import *

# function to create linkage matrix with columns required for further functions
def create_linkage_matrix(projection_df):
    linkage_matrix_np = linkage(projection_df, method='ward')
    linkage_matrix = pd.DataFrame(linkage_matrix_np, columns=['cluster_index_1', 'cluster_index_2', 'distance', 'samples'])
    return linkage_matrix

# function to plot dendrogram at given level and color threshold
def plot_dendrogram(linkage_matrix, level, orientation='right', color_threshold=None, save_path=None):
    '''
    Parameters
    ----------
    linkage_matrix : pandas DataFrame
        linkage matrix
    level : integer
        Number of clusters to end at, use 0 for full dendrogram
    orientation : string, optional
        DESCRIPTION. The default is 'right'.
    color_threshold : double, optional
        DESCRIPTION. clusters branches with distances below this will appear in different color schematic
    save_path : string, optional
        DESCRIPTION. The default is None, which will produce just the plot. Else will attempt to save in given directory path.

    Returns
    -------
    None.

    '''
    plt.figure(figsize=(12, 8))
    dendrogram(linkage_matrix,
               p=level,
               truncate_mode='level',
               orientation=orientation,
               color_threshold=color_threshold)
    if level == 0:
        plt.title('Full Dendrogram')
    elif (level == 0) and (color_threshold):
        plt.tile(f'Full Dendrogram with Threshold {color_threshold}')
    elif color_threshold:
        plt.title(f'Dendrogram Truncated at Level {level} with Threshold {color_threshold}')
    else:
        plt.title(f'Dendrogram Truncated at Level {level}')
    plt.xlabel('Distance')
    plt.ylabel('Leaves')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
# function to retrieve dendrogram information
def retrieve_dendrogram_info(linkage_matrix, color_threshold=None, level=None):
    if level:
        dendrogram_info = dendrogram(linkage_matrix,
                                     p=level,
                                     distance_sort='descending',
                                     truncate_mode='level',
                                     color_threshold=color_threshold,
                                     no_plot=True)
    else:
        dendrogram_info = dendrogram(linkage_matrix,
                                     distance_sort='descending',
                                     color_threshold=color_threshold,
                                     no_plot=True)
    
    return dendrogram_info

# function to track merges within a linkage matrix
def track_merges(linkage_matrix):
    # first merge forms cluster number {original_samples}
    clusters_formed = linkage_matrix.shape[0] + 1
    cluster_formation = {'cluster_index_1': [],
                         'cluster_index_2': [],
                         'cluster_formed': [],
                         'samples': []}
    
    for index, row in linkage_matrix.iterrows():
        # get information
        cluster_index_1 = row['cluster_index_1']
        cluster_index_2 = row['cluster_index_2']
        samples = row['samples']
        cluster_formed = clusters_formed
        clusters_formed += 1
        
        # add to data structure
        cluster_formation['cluster_index_1'].append(int(cluster_index_1))
        cluster_formation['cluster_index_2'].append(int(cluster_index_2))
        cluster_formation['cluster_formed'].append(int(cluster_formed))
        cluster_formation['samples'].append(int(samples))
        
    return pd.DataFrame(cluster_formation)

# function to trace parents of clustes (agglomerative is bottom-up)
def trace_parents(cluster_formation):
    original_samples = cluster_formation.shape[0]
    
    parent_keys = dict()
    for index, row in cluster_formation.iterrows():
        cluster_formed = row['cluster_formed']
        parent_1 = row['cluster_index_1']
        parent_2 = row['cluster_index_2']
        parents = [parent_1, parent_2]
        if row['samples'] > 2:
            try:
                linked_keys_1 = parent_keys[parent_1]
                parents = parents + linked_keys_1
            except:
                parents = parents
            try:
                linked_keys_2 = parent_keys[parent_2]
                parents = parents + linked_keys_2
            except:
                parents = parents
                
        parent_keys[cluster_formed] = parents
        
    parent_nodes = dict()
    parent_clusters = dict()
    
    for parent in parent_keys:
        parent_list = parent_keys[parent]
        parent_nodes[parent] = [parent for parent in parent_list if parent <= original_samples]
        parent_clusters[parent] = [parent for parent in parent_list if parent > original_samples]
        
    return parent_keys, parent_nodes, parent_clusters

# function to expand leaves on a non-full dendrogram
def expand_leaves(projection_df, level, color_threshold=None):
    # run functions for necessary inputs
    linkage_matrix = create_linkage_matrix(projection_df)
    dendrogram_info = retrieve_dendrogram_info(linkage_matrix, color_threshold=color_threshold, level=level)
    cluster_formation = track_merges(linkage_matrix)
    parent_trace = trace_parents(cluster_formation)[1]
    
    parent_nodes = {'leaf_order': [],
                    'leaf': [],
                    'ivl': [],
                    'parent_nodes': [],
                    'cluster': []}
    leaf_order = 0
    for leaf, ivl, cluster in zip(dendrogram_info['leaves'], dendrogram_info['ivl'], dendrogram_info['leaves_color_list']):
        parent_nodes['leaf_order'].append(leaf_order)
        parent_nodes['leaf'].append(leaf)
        parent_nodes['ivl'].append(ivl)
        parent_nodes['cluster'].append(cluster)
        try:
            parent_nodes['parent_nodes'].append(parent_trace[leaf])
        except:
            parent_nodes['parent_nodes'].append([leaf])
            
        leaf_order += 1
    
    return pd.DataFrame(parent_nodes)

# function to get label spread statistics by leaves
def analyze_leaves(results_df, parent_nodes, label_col, save_path=None, return_df=None):
    full_df = results_df.copy()
    # get indices for each cluster
    cluster_indices = {cluster: [] for cluster in parent_nodes['cluster'].unique()}
    for index, row in parent_nodes.iterrows():
        cluster = row['cluster']
        parents = row['parent_nodes']
        cluster_indices[cluster] += parents
    
    # create melted dataframe with the clusters and indices
    cluster_df = pd.DataFrame(dict([(cluster, pd.Series(parents)) for cluster, parents in cluster_indices.items()]))
    melted_df = cluster_df.melt(var_name='cluster', value_name='parent').dropna()
    melted_df['parent'] = melted_df['parent'].astype(int)
    melted_df.set_index('parent', inplace=True)
    full_df = full_df.merge(melted_df, left_index=True, right_index=True)
    
    # create stacked barplots
    # crosstab creates a table showing the frequency distribution of clusters and label_col
    crosstab = pd.crosstab(full_df['cluster'], full_df[label_col])
    ax = crosstab.plot(kind='bar', stacked=True, colormap='viridis')
    plt.title(f'{label_col.title()} Spread by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.legend(title=label_col.title(), bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticklabels(range(1, len(crosstab.index) + 1))
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # create chi-squared contingency table
    '''
    - chi2: chi-square statistic: measure the difference between observed and expected frequencies
    - p: if this value is less than significance level -> reject null hypothesis -> indicates significant association between clusters and col_label
    - dof: number of independent values or quantities which can be assigned to a statistical distribution
    - expected: frequences that would be expected if there was no association between the variables
    '''
    chi2, p, dof, expected = chi2_contingency(crosstab)
    
    if p < 0.05:
        print(f'\nP-VALUE: {p}\nHYPOTHESIS: reject null hypothesis\nCONCLUSION: indicates significant association between clusters and {label_col.lower()}')
    else:
        print(f'\np-value: {p}\nHYPOTHESIS: fail to reject null hypothesis\nCONCLUSION: suggests there is evidence that there is not significant association between clusters and {label_col.lower()}')
    
    if return_df:
        return full_df
