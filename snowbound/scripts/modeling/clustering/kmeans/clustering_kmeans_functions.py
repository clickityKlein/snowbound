'''
KMEANS CLUSTERING FUNCTIONS
'''

'''
Imports
'''
## LIBRARY IMPORTS ##
# standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import os
import sys
from scipy.spatial.distance import cdist
import importlib.util

# plotly 3d visualization
import plotly.express as px
import plotly.graph_objects as go

# sklearn libraries
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from sklearn.utils import resample

# custom pca functions
module_name = 'pca_functions'
module_path = os.path.join(os.getcwd(), '../pca/pca_functions.py')
spec = importlib.util.spec_from_file_location(module_name, module_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)

# custom pca functions
sys.path.insert(0, '../pca')
from pca_functions import *

'''
Clustering with Silhouette and Aided by Elbow
'''
# function to create a silhouette plot
def create_silhoutte_plot(df_decision, n_clusters, avg_silhouette_score, sample_silhouette_values, silhouette_save_path=None):
    '''
    Function to create a silhouette plot based on PCA projection.

    Parameters
    ----------
    df_decision : pandas DataFrame
        Either a direct dataframe with PCA projected data or a subset of that.
    n_clusters : int
        Number of clusters for KMeans.
    avg_silhouette_score : float
        Average Silhouette Score of the model.
    sample_silhouette_values : numpy Array
        Silhouette Coefficients of the model.
    silhouette_save_path : string, optional
        If save, saves in provided folder. The default is None.

    Returns
    -------
    None.

    '''
    
    # df_decision: either df_projection or model_subset
    
    # initialize plot
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(18, 7)

    # Set the limits for the silhouette plot
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, df_decision.shape[0] + (n_clusters + 1) * 10])
    y_lower = 10
    
    # create silhouette plot for each cluster group
    for cluster_group in range(n_clusters):
        group_cluster_silhouette_values = sample_silhouette_values[df_decision['Cluster'] == cluster_group]
        group_cluster_silhouette_values.sort()

        size_cluster_group = group_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_group

        color = cm.nipy_spectral(float(cluster_group) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, group_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_group, str(cluster_group))

        y_lower = y_upper + 10

    ax1.set_title(f'Silhouette Plot for {n_clusters} Clusters')
    ax1.set_xlabel('Silhouette Coefficient Values')
    ax1.set_ylabel('Cluster Label')

    ax1.axvline(x=avg_silhouette_score, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks([x / 10.0 for x in range(-1, 11)])
    
    # save option
    if silhouette_save_path:
        plt.savefig(silhouette_save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
# function to test cluster groupings with the silhouette method
def silhouette_kmeans(df_projection, cluster_min, cluster_max, silhouette_path_start, cluster_path_start, plot_names, sample_size=None):
    '''
    Function to aid in analyzing sihouette plots for a range of clustering values.

    Parameters
    ----------
    df_projection : pandas DataFrame
        Data projected onto a PCA space.
    cluster_min : int
        Minimum number of clusters to test.
    cluster_max : int
        Maximum number of clusters to test.
    silhouette_path_start : str
        Start path of where to save silhouette plots.
    cluster_path_start : str
        Start path of where to save cluster plots.
    plot_names : str
        Additional component for image save paths.
    sample_size : float, optional
        Sample size of PCA projection for reduction purposes. The default is None.

    Returns
    -------
    silhouette_results : dictionary
        Results of the silhouette analysis.

    '''
    
    # end immediately if path does not exist
    if not os.path.exists(silhouette_path_start):
        print(f'silhouette_path_start: "{silhouette_path_start}" does not exist')
        return None
    
    if not os.path.exists(cluster_path_start):
        print(f'cluster_path_start: "{cluster_path_start}" does not exist')
        return None
    
    # retain to return silhouette components
    silhouette_results = {'clusters': [],
                          'sil_averages': [],
                          'sil_samples': []}
    
    for cluster in range(cluster_min, cluster_max+1):
        # create kmeans clustering model
        model = KMeans(n_clusters=cluster)
        # fit model
        model.fit(df_projection)
        # predict with model (label / cluster)
        model_labels = model.predict(df_projection)
        # concatenate projection with labels
        df_decision = pd.concat([df_projection, pd.Series(model_labels)], axis=1)
        df_decision.rename(columns={0:'Cluster'}, inplace=True)
        
        # if sample_size is not None -> will take a subset of given size
        if sample_size:
            # create the subset based on sample_size
            df_decision = resample(df_decision, n_samples=int(df_decision.shape[0] * sample_size), random_state=42)
        
        # get silhouette information
        avg_silhouette_score = silhouette_score(df_decision[['principal_component_1', 'principal_component_2', 'principal_component_3']], df_decision['Cluster'])
        sample_silhouette_values = silhouette_samples(df_decision[['principal_component_1', 'principal_component_2', 'principal_component_3']], df_decision['Cluster'])
        
        # put data into dictionary
        silhouette_results['clusters'].append(cluster)
        silhouette_results['sil_averages'].append(avg_silhouette_score)
        silhouette_results['sil_samples'].append(sample_silhouette_values)
        
        # silhouette save path
        silhouette_save_path = f'{silhouette_path_start}_{plot_names}_clusters_{cluster}.png'
        
        # create silhouette plot
        create_silhoutte_plot(df_decision=df_decision,
                              n_clusters=cluster,
                              avg_silhouette_score=avg_silhouette_score,
                              sample_silhouette_values=sample_silhouette_values,
                              silhouette_save_path=silhouette_save_path)
        
        # cluster save path
        cluster_save_path = f'{cluster_path_start}_{plot_names}_clusters_{cluster}.html'
        
        # create cluster plot
        visualize_results_3d(df_decision, 'Cluster', 'Cluster', cluster_save_path, opacity=0.7, arrow_size=10)
        
        # print progress
        print(f'Cluster Tested: {cluster}')
        
    # return results dictionary
    return silhouette_results

# function to plot average silhouettes
def plot_average_silhouette(average_list, cluster_min, cluster_max, save_path, plot_name):
    '''
    Create plot to analyze the average silhouette scores for each clustering specification.

    Parameters
    ----------
    average_list : list
        List of the average silhouette scores.
    cluster_min : int
        Minimum number of clusters.
    cluster_max : int
        Maximum number of clusters.
    save_path : string
        Base directory for image pathing.
    plot_name : string
        Illustrative and saving purposes.

    Returns
    -------
    None.

    '''
    
    plt.plot(range(cluster_min, cluster_max+1), average_list)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Coefficient')
    plt.title(f'Average Silhouette Scores - {plot_name.title()}')
    plt.savefig(f'{save_path}/coefficients_{plot_name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
# function to create dataframe for silhouette samples
def silhouette_to_df(silhouette_results):
    df = pd.DataFrame(silhouette_results['sil_samples']).T
    df.columns = silhouette_results['clusters']
    
    return df
        
# function to plot elbow method in k-means
def plot_elbow(wcss, cluster_min, cluster_max, save_path, plot_name):
    plt.plot(range(cluster_min, cluster_max+1), wcss)
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.title(f'Elbow Method - {plot_name.title()}')
    plt.savefig(f'{save_path}/elbow_{plot_name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
def elbow_kmeans(df_projection, cluster_min, cluster_max, elbow_path_start, plot_name):
    wcss_results = {'clusters': [],
                    'wcss': []}
    wcss= []
    for cluster in range(cluster_min, cluster_max+1):
        model = KMeans(n_clusters=cluster)
        y_kmeans = model.fit_predict(df_projection)
        wcss.append(model.inertia_)
        wcss_results['clusters'].append(cluster)
    
    # plot
    plot_elbow(wcss, cluster_min, cluster_max, elbow_path_start, plot_name)
    
    # put final wcss into results
    wcss_results['wcss'] = wcss
    
    # return wcss results
    return wcss_results

def create_sphere(center, radius=1, resolution=20):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z
    
def create_circle(center, radius=1, resolution=20):
    theta = np.linspace(0, 2 * np.pi, resolution)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    z = np.full_like(x, center[2])
    return x, y, z

def get_radius(df_complete, centroids, clusters):
    radius_results = dict()
    for cluster in range(clusters):
        cluster_coordinates = np.array(df_complete[df_complete['cluster_label']==cluster][['principal_component_1', 'principal_component_2', 'principal_component_3']])
        distances = cdist(centroids[cluster].reshape(1, -1), cluster_coordinates, metric='euclidean')
        radius_results[cluster] = np.max(distances)
    
    return radius_results

def visualize_clustering_results(df_results, clusters, label_col, legend_title, main_name, save_path, opacity=0.7, resolution=100, circles=True, subset=None):
    # create a copy of df_results
    df_results = df_results.copy()
    # perform kmeans clustering
    model = KMeans(n_clusters=clusters)
    # fit model on pca space
    model.fit(df_results[['principal_component_1', 'principal_component_2', 'principal_component_3']])
    # predict model and get labels
    model_labels = model.predict(df_results[['principal_component_1', 'principal_component_2', 'principal_component_3']])
    # append model cluser labels to df
    df_complete = pd.concat([df_results, pd.DataFrame(model_labels, columns=['cluster_label'])], axis=1)
    if subset is not None:
        df_complete = df_complete.sample(int(df_complete.shape[0] * subset))
        df_complete.reset_index(drop=True, inplace=True)
        
    # find model centroids
    centroids = model.cluster_centers_
    # get max radii
    max_radii = get_radius(df_complete, centroids, clusters)
    
    # begin figure
    fig = px.scatter_3d(df_complete, x='principal_component_1', y='principal_component_2', z='principal_component_3', color=label_col, opacity=opacity)
    
    # create centroids trace
    centroids_trace = go.Scatter3d(
        x=centroids[:, 0],
        y=centroids[:, 1],
        z=centroids[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color='black',
            symbol='x'
            ),
        name='Centroids'
        )
    
    # add centroids trace
    fig.add_trace(centroids_trace)
    
    # add circles around centroids
    if circles:
        for cluster_num, centroid in enumerate(centroids):
            x, y, z = create_circle(centroid, radius=max_radii[cluster_num], resolution=resolution)
            
            fig.add_trace(go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='lines',
                line=dict(color='black'),
                name=f'Cluster {cluster_num + 1} Radius'
            ))
    
    # disable layout panes
    fig.update_layout(scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False)
    ))
    
    # update legend
    fig.update_layout(legend=dict(
        title=legend_title,
        x=0.1,  # Position of the legend
        y=0.9,
        bgcolor='rgba(255, 255, 255, 0.5)',  # Background color with transparency
        bordercolor='black',
        borderwidth=2
    ))
    
    # write plotly figure to html
    fig.write_html(f'{save_path}/{main_name}_{label_col}_{clusters}.html')
