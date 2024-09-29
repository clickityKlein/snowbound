'''
PCA - FUNCTIONS
'''

'''
Imports
'''
## LIBRARY IMPORTS ##
# standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# plotly 3d visualization
import plotly.express as px
import plotly.graph_objects as go

# sklearn libraries
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

## DATA IMPORTS ##
'''
# relative path to data
relative_path = '../../data'

# weather
weather_path = os.path.join(relative_path, 'weather_cleaned.csv')
weather = pd.read_csv(weather_path)
weather['datetime'] = pd.to_datetime(weather['datetime'])

# google places
google_path = os.path.join(relative_path, 'google_cleaned.csv')
google = pd.read_csv(google_path)

# resorts
resort_path = os.path.join(relative_path, 'resort_cleaned.csv')
resorts = pd.read_csv(resort_path)
'''

'''
Functions
'''
## FUNCTIONS ##
# function to perform PCA
def perform_pca(df, quant_cols, label_cols, n_components=None):
    '''
    Perform Principal Component Analysis (PCA) on a given DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the data to be analyzed.
    quant_cols : list of str
        List of column names in `df` that contain quantitative data.
    label_cols : list of str
        List of column names in `df` that contain label data.
    n_components : int, optional
        Number of principal components to compute. If None, all components are computed.

    Returns:
    --------
    df_projection : pandas.DataFrame
        DataFrame containing the projection of the data into the PCA space.
    df_explained : pandas.DataFrame
        DataFrame containing the explained variance and cumulative variance for each principal component.
    df_components : pandas.DataFrame
        DataFrame containing the principal components (eigenvectors).
    df_loadings : pandas.DataFrame
        DataFrame containing the loadings matrix.
    df_results : pandas.DataFrame
        DataFrame containing the PCA space projection along with the original label column(s).
    df_eigen : pandas.DataFrame
        Dataframe containing the actual eigenvalues from the pca process
    
    Notes:
    ------
    - The function normalizes the quantitative data using `StandardScaler` before applying PCA.
    - The explained variance and cumulative variance provide insights into the amount of variance captured by each principal component.
    - The principal components (eigenvectors) indicate the direction of maximum variance in the data.
    - The loadings matrix represents the correlation between the original variable and principal component.
    - Eigenvalues will be printed as well.
    '''
    
    # separate quantitative columns and label column(s)
    df_quant = df[quant_cols]
    df_label = df[label_cols]
    
    # normalize via StandardScaler (from sklearn.preprocessing import StandardScaler)
    scaler = StandardScaler()
    df_normal = scaler.fit_transform(df_quant)
    
    # create PCA object (from sklearn.decomposition import PCA)
    # components or general
    if n_components:
        pca = PCA(n_components=n_components)
    else:
        pca = PCA()
    
    # fit and transform PCA object to get the components - project into PCA space
    pca_projection = pca.fit_transform(df_normal)
    
    # dataframe for PCA space projection
    df_projection = pd.DataFrame(pca_projection)
    df_projection.columns = [f'principal_component_{col+1}' for col in range(df_projection.shape[1])]
    
    # dataframe for explained variance ratio - eigenvalues' ratio, include cumulative variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    df_explained = pd.DataFrame({
            'principal_components': [f'principal_component_{col+1}' for col in range(len(explained_variance))],
            'explained_variance': explained_variance,
            'cumulative_variance': cumulative_variance
        })
    
    # get actual eigenvalues
    eigenvalues = pca.explained_variance_
    print(f'Eigenvalues Results:\n{eigenvalues}')
    df_eigen = pd.DataFrame(eigenvalues, columns=['Eigenvalue'])
    df_eigen.index = [f'Principal Component {n+1}' for n in df_eigen.index]
    
    # dataframe for PCA components - eigenvectors
    space_components = pca.components_.T
    df_components = pd.DataFrame(space_components)
    df_components.columns = [f'principal_component_{col}' for col in range(df_components.shape[1])]
    
    # dataframe for loadings matrix
    df_loadings = pd.DataFrame(pca.components_.T, columns=[f'principal_component_{col+1}' for col in range(pca.components_.shape[0])], index=df_quant.columns)
    
    # dataframe in PCA space and labels
    df_results = pd.concat([df_projection, df_label], axis=1)
    
    
    # return all results
    return df_projection, df_explained, df_components, df_loadings, df_results, df_eigen

# function to validate orthogonality of eigenvector return (df_components)
def validate_orthogonality(df_components):
    '''
    Parameters
    ----------
    df_components : pandas.DataFrame
        Result from `perform_pca`, which are the eigenvectors of a PCA transformation.

    Returns
    -------
    orthogonality_validate : pandas.DataFrame
        Dot product of `df_components` and transposed `df_components`.
    
    Notes:
    ------
    - Expected return is 1's along the left-right diagonal and very small numbers elsewhere (simulated zero).
    '''
    
    # dot product of eigenvalues' matrix and transposed eigenvalues' matrix 
    orthogonality_validate = df_components.T.dot(df_components)
    
    # return
    return orthogonality_validate

# function to plot explained and cumulative variance
def visualize_variance(df_explained, title_suffix=None, threshold=None, save_path=None):
    '''
    Visualizes the explained and cumulative variance by principal components.

    Parameters
    ----------
    df_explained : pandas.DataFrame 
        Result from `perform_pca`. A DataFrame containing the explained and cumulative variance data.
    title_suffix: str, optional
        Ending for the plot title, can specify dataset, etc.
    threshold : float, optional
        A threshold value for cumulative variance. Must be between 0 and 1. If provided, a horizontal line is drawn at this value.
    save_path: str, optional
        The file path to save the plot. If provided, the plot is saved to this path.

    Returns:
    -------
    None
    
    Notes:
    ------
    - This function creates a bar plot for the explained variance and a line plot for the cumulative variance by principal components.
    - If a threshold is specified, it adds a horizontal line at the threshold value and labels the principal component where the threshold is met.
    - The plot can be saved to a specified path if the save_path parameter is provided.
    '''
    
    # threshold error handling
    if threshold is not None:
        if not isinstance(threshold, float) or not (0 <= threshold <= 1):
            raise ValueError('Threshold must be a float between 0 and 1.')
    
    # initialize figure
    plt.figure(figsize=(12, 8))
    # base color blue
    base_color = sns.color_palette()[0]
    # barplot for explained variance by principal component
    sns.barplot(data=df_explained, y='explained_variance', x='principal_components', color=base_color, label='Explained Variance')
    # lineplot for cumulative explained variance by principal component
    sns.lineplot(data=df_explained, y='cumulative_variance', x='principal_components', color='red', label='Cumulative Variance')
    
    # treshold option
    if threshold:
        # get principal component - threshold intersect
        try:
            pc_intersect = df_explained.loc[df_explained['cumulative_variance']>=threshold, 'principal_components'].index[0]
            plt.axhline(y=threshold, color='black', linestyle='--', label=f'{threshold:.1%} Information Retention at Component {pc_intersect+1}')
        except:
            raise IndexError('Threshold Greater than Possible Cumulative Variance.')
    
    # additional touches
    # title option
    if title_suffix:
        plt.title(f'Individual Explained Variance by Principal Components - {title_suffix}')
    else:
        plt.title('Individual Explained Variance by Principal Components')
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance')
    plt.xticks(range(df_explained.shape[0]), range(1, df_explained.shape[0] + 1))
    plt.legend(loc='center right')
    plt.grid(True)
    
    # save option
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
# function for further analysis on the loadings matrix
def melt_loadings(df_loadings, use_rank=False):
    '''
    Parameters
    ----------
    df_loadings : pandas.DataFrame
        Result from `perform_pca`. A DataFrame containing representing the correlation between the original variable and principal component.
    use_rank : bool, optional
        The default is False. Returns raw correlations. If True, returns absolute correlation rankings.

    Returns
    -------
    df_long : pandas.DataFrame
        Melted loadings matrix, either by raw correlations or ranks.
        
    Notes:
    ------
    - Given raw correlations, useful in:
        - sns.boxplot(data=df_long, x='correlation', y='quantitative_variables', color=base_color)
        - sns.boxplot(data=df_long, x='correlation', y='principal_components', color=base_color)
    - Given rankings, useful in:
        - sns.boxplot(data=df_long, x='rank', y='quantitative_variables', color=base_color)
        - df_avg = df_long.groupby('quantitative_variables')['rank'].mean().reset_index().sort_values(by='rank')
        - sns.barplot(data=df_avg, y='quantitative_variables', x='rank', color=base_color)
    '''
    
    if use_rank:
        df_ranks = df_loadings.abs().rank(ascending=False)
        df_long = df_ranks.reset_index().melt(id_vars='index', var_name='principal_components', value_name='rank')
        df_long.rename(columns={'index': 'quantitative_variables'}, inplace=True)
    else:
        df_long = df_loadings.reset_index().melt(id_vars='index', var_name='principal_components', value_name='correlation')
        df_long.rename(columns={'index': 'quantitative_variables'}, inplace=True)
        
    return df_long

# function to produce barplot of loadings matrix rankings
def rank_and_visualize_loadings(df_loadings, plot_type='barplot', save_path=None):
    '''
    Parameters
    ----------
    df_loadings : pandas.DataFrame
        Correlation of each quantitative variable to the prinicpal components.
    plot_type: str, optional
        Default is barplot. Specify barplot or boxplot.
    save_path : str, optional
        The file path to save the plot. If provided, the plot is saved to this path.

    Returns
    -------
    None.
    
    Notes:
    ------
    - Creates visual of the average ranking of variables across principal components.

    '''
    
    # base color
    base_color = sns.color_palette()[0]
    
    # create long
    df_long = melt_loadings(df_loadings.abs(), use_rank=True)
    
    # average
    df_avg = df_long.groupby('quantitative_variables')['rank'].mean().reset_index().sort_values(by='rank')
    
    if plot_type == 'barplot':
        # initialize and create figure
        plt.figure(figsize=(12, 8))
        sns.barplot(data=df_avg, y='quantitative_variables', x='rank', color=base_color)
        plt.title('Average Correlation Rank of Principal Components by Variable')
        plt.xlabel('Average Rank')
        plt.ylabel('Quantitative Variables')
        
    elif plot_type == 'boxplot':
        # initialize and create figure
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df_long, x='rank', y='quantitative_variables', color=base_color, order=df_avg['quantitative_variables'])
        plt.title('Spread of Correlation Rank of Principal Components by Variable')
        plt.xlabel('Rank')
        plt.ylabel('Quantitative Variables')
    
    # save option
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
# function to visualize 3d results in a static frame
def visualize_results_3d(df_results, label_col, legend_title, save_path, opacity=0.7, arrow_size=10):
    '''
    Parameters
    ----------
    df_results : pandas.DataFrame
        DataFrame containing the PCA space projection along with the original label column(s).
    label_col : str
        String for the column name representing the label of the data.
    legend_title : str
        Title type string for the legend title.
    save_path : str
        The file path to save the plot. Depending on IDE and settings, likely best to view html visual object in browser.
    opacity : float, optional
        Opacity of projected data points. The default is 0.7.
    arrow_size : int, optional
        Axis length for each principal component. The default is 10.

    Returns
    -------
    None.

    '''
    
    # begin figure
    fig = px.scatter_3d(df_results, x='principal_component_1', y='principal_component_2', z='principal_component_3', color=label_col, opacity=opacity)
    
    # arrow options (principal component axes)
    arrows = [
        go.Scatter3d(x=[-arrow_size, arrow_size], y=[0, 0], z=[0, 0], mode='lines+text', line=dict(color='black', width=8), text=['PC1 (-)', 'PC1 (+)'], textposition='top center', showlegend=False),
        go.Scatter3d(x=[0, 0], y=[-arrow_size, arrow_size], z=[0, 0], mode='lines+text', line=dict(color='black', width=8), text=['PC2 (-)', 'PC2 (+)'], textposition='top center', showlegend=False),
        go.Scatter3d(x=[0, 0], y=[0, 0], z=[-arrow_size, arrow_size], mode='lines+text', line=dict(color='black', width=8), text=['PC3 (-)', 'PC3 (+)'], textposition='top center', showlegend=False)
    ]

    # add arrow traces
    fig.add_traces(arrows)
    
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
    fig.write_html(save_path)

# function to visualize 3d results in an animation frame
def animate_results_timeseries_3d(df_results, label_col, legend_title, save_path, time_col='datetime', opacity=0.7, arrow_size=10):
    '''
    Parameters
    ----------
    df_results : pandas.DataFrame
        DataFrame containing the PCA space projection along with the original label column(s).
    label_col : str
        String for the column name representing the label of the data.
    legend_title : str
        Title type string for the legend title.
    save_path : str
        The file path to save the plot. Depending on IDE and settings, likely best to view html visual object in browser.
    time_col : str, optional
        String for the column name representing the time column of the data. The default is 'datetime'.
    opacity : float, optional
        Opacity of projected data points. The default is 0.7.
    arrow_size : int, optional
        Axis length for each principal component. The default is 10.

    Returns
    -------
    None.

    '''
    
    # create copy of df_results and add column for datetime animation
    df = df_results.copy()
    df['date'] = df[time_col].dt.strftime('%Y-%m')

    # Create an animation
    fig = px.scatter_3d(df, 
                        x='principal_component_1', 
                        y='principal_component_2', 
                        z='principal_component_3', 
                        color=label_col,
                        opacity=opacity,
                        animation_frame='date',
                        labels={'date': 'Date'}
                        )
        
    # Adding axis arrows
    arrows = [
        go.Scatter3d(x=[-10, 10], y=[0, 0], z=[0, 0], mode='lines+text', line=dict(color='black', width=8), text=['PC1 (-)', 'PC1 (+)'], textposition='top center', showlegend=False),
        go.Scatter3d(x=[0, 0], y=[-10, 10], z=[0, 0], mode='lines+text', line=dict(color='black', width=8), text=['PC2 (-)', 'PC2 (+)'], textposition='top center', showlegend=False),
        go.Scatter3d(x=[0, 0], y=[0, 0], z=[-10, 10], mode='lines+text', line=dict(color='black', width=8), text=['PC3 (-)', 'PC3 (+)'], textposition='top center', showlegend=False)
    ]

    fig.add_traces(arrows)

    # Removing gray planes and axes
    fig.update_layout(scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False)
    ))

    # Updating the legend
    fig.update_layout(legend=dict(
        title=legend_title,
        x=0.1,  # Position of the legend
        y=0.9,
        bgcolor='rgba(255, 255, 255, 0.5)',  # Background color with transparency
        bordercolor='black',
        borderwidth=2
    ))

    fig.write_html(save_path)

# function to visualize 2d results in a static frame
def visualize_results_2d(df_results, label_col, legend_title, title_suffix=None, save_path=None):
    '''
    Parameters
    ----------
    df_results : pandas.DataFrame
        DataFrame containing the PCA space projection along with the original label column(s).
    label_col : str
        String for the column name representing the label of the data.
    legend_title : str
        Title type string for the legend title.
    title_suffix: str, optional
        Ending for the plot title, can specify dataset, etc. The default is None.
    save_path : str
        The file path to save the plot. The default is None.

    Returns
    -------
    None.

    '''
    
    # initialize figure
    plt.figure(figsize=(12, 8))
    
    # visualize 2d results
    sns.scatterplot(data=df_results, x='principal_component_1', y='principal_component_2', hue=label_col)
    plt.axhline(y=0, color='black')
    plt.axvline(x=0, color='black')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    
    # title option
    if title_suffix:
        plt.title(f'2 Dimensional PCA - {title_suffix}')
    else:
        plt.title('2 Dimensional PCA')
    
    plt.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # save option
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    # plot
    plt.show()
