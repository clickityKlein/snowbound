from flask import render_template, redirect, url_for, request, Blueprint, flash, send_from_directory
import pandas as pd
import os

model_pages_controls = Blueprint('model_pages_controls', __name__)

# function to aid in pca page output
def formulate_pca_output(dataset, dimension):
    # path start
    adv_path_start = 'snowbound/static/models/pca/'
    img_path_start = '../../static/models/pca'
    html_path_start = 'models/pca'
    
    # explained variance df
    explained_variance_df = pd.read_csv(os.path.join(adv_path_start, f'explained_{dataset}_{dimension}.csv'))
    # reformatting explained variance df
    explained_variance_df.drop(columns=['Unnamed: 0'], inplace=True)
    percent_cols = ['explained_variance', 'cumulative_variance']
    explained_variance_df[percent_cols] = explained_variance_df[percent_cols].map(lambda x: '{:.2%}'.format(x))
    explained_variance_df = explained_variance_df.to_html(classes='table table-striped', index=False, justify='left')
    
    # loadings df
    loadings_df = pd.read_csv(os.path.join(adv_path_start, f'loadings_{dataset}_{dimension}.csv'))
    # reformatting loadings df
    loadings_df.rename(columns={'Unnamed: 0': 'Feature'}, inplace=True)
    loadings_df = loadings_df.to_html(classes='table table-striped', index=False, justify='left')
    
    # loadings barplot png
    loadings_bar_png = os.path.join(img_path_start, f'loadings_matrix_rank_barplot_{dataset}_{dimension}.png')
    
    # loadings boxplot png
    loadings_box_png = os.path.join(img_path_start, f'loadings_matrix_rank_boxplot_{dataset}_{dimension}.png')
    
    
    if dimension=='gen':
        # visualize variance png
        explained_variance_png = os.path.join(img_path_start, f'explained_variance_{dataset}_{dimension}_95.png')
        
        # top eigenvalues df
        top_eigens_df = pd.read_csv(os.path.join(adv_path_start, f'eigen_{dataset}_{dimension}.csv'))
        # reformatting eigenvalues df
        top_eigens_df.rename(columns={'Unnamed: 0': 'Principal Component'}, inplace=True)
        top_eigens_df = top_eigens_df.to_html(classes='table table-striped', index=False, justify='left')
        
        # projection
        projection = None
        
    elif dimension=='3d':
        # visualize variance png
        explained_variance_png = os.path.join(img_path_start, f'explained_variance_{dataset}_{dimension}.png')
        
        # top eigenvalues
        top_eigens_df = None
        
        # projection: animated for weather, triple for resort, single for google
        if dataset=='weather':
            # animated plot
            projection = os.path.join(html_path_start, 'weather_3d_animation.html')
        elif dataset=='resorts':
            # triple for resort
            projection = {0: os.path.join(html_path_start, 'resorts_country_3d_vis.html'),
                          1: os.path.join(html_path_start, 'resorts_pass_3d_vis.html'),
                          2: os.path.join(html_path_start, 'resorts_region_3d_vis.html')}
        else:
            # non-animated plot
            projection = os.path.join(html_path_start, f'{dataset}_3d_vis.html')
            
    else:
        # visualize variance png
        explained_variance_png = os.path.join(img_path_start, f'explained_variance_{dataset}_{dimension}.png')
        
        # top eigenvalues
        top_eigens_df = None
        
        # projection plot: triple for resort, single for others
        if dataset=='resorts':
            projection = {0: os.path.join(img_path_start, 'resorts_country_2d_vis.png'),
                          1: os.path.join(img_path_start, 'resorts_pass_2d_vis.png'),
                          2: os.path.join(img_path_start, 'resorts_region_2d_vis.png')}
        else:
            projection = os.path.join(img_path_start, f'{dataset}_2d_vis.png')
            
    # returns
    return explained_variance_df, explained_variance_png, top_eigens_df, loadings_df, loadings_bar_png, loadings_box_png, projection

# render pca model page
@model_pages_controls.route('/model_pages/model_pca.html', methods=['GET', 'POST'])
def load_pca_page():
    # final resort data - snippet
    df = pd.read_csv('snowbound/data/cleaned_data/resorts_snippet.csv')
    resorts_final = df.to_html(classes='table table-striped', index=False)
    
    # weather final - snippet
    df = pd.read_csv('snowbound/data/cleaned_data/weather_snippet.csv')
    weather_final = df.to_html(classes='table table-striped', index=False)
    
    # google types final - snippet
    df = pd.read_csv('snowbound/data/cleaned_data/google_places_snippet.csv')
    google_final = df.to_html(classes='table table-striped', index=False)
    
    # create output
    datasets = ['resorts', 'weather', 'google']
    dimensions = ['gen', '3d', '2d']
    data_dim = dict()
    
    for dataset in datasets:
        for dimension in dimensions:
            data_dim[f'{dataset}_{dimension}'] = formulate_pca_output(dataset, dimension)
    
    return render_template('model_pages/model_pca.html',
                           resorts_final=resorts_final,
                           weather_final=weather_final,
                           google_final=google_final,
                           data_dim=data_dim)

# render clustering model page
@model_pages_controls.route('/model_pages/model_clustering.html', methods=['GET', 'POST'])
def load_clustering_page():
    return render_template('model_pages/model_clustering.html')

# render arm model page
@model_pages_controls.route('/model_pages/model_arm.html', methods=['GET', 'POST'])
def load_arm_page():
    return render_template('model_pages/model_arm.html')

# render dt model page
@model_pages_controls.route('/model_pages/model_dt.html', methods=['GET', 'POST'])
def load_dt_page():
    return render_template('model_pages/model_dt.html')

# render nb model page
@model_pages_controls.route('/model_pages/model_nb.html', methods=['GET', 'POST'])
def load_nb_page():
    return render_template('model_pages/model_nb.html')

# render svm model page
@model_pages_controls.route('/model_pages/model_svm.html', methods=['GET', 'POST'])
def load_svm_page():
    return render_template('model_pages/model_svm.html')

# render regression model page
@model_pages_controls.route('/model_pages/model_regression.html', methods=['GET', 'POST'])
def load_regression_page():
    return render_template('model_pages/model_regression.html')
