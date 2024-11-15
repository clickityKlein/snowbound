from flask import render_template, redirect, url_for, request, Blueprint, flash, send_from_directory
import pandas as pd
import os

model_pages_controls = Blueprint('model_pages_controls', __name__)

'''
PCA
'''
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

# function to aid in kmeans clustering page output
def formulate_clustering_output_kmeans():
    # path start
    # adv_path_start = 'snowbound/static/models/clustering/kmeans'
    img_path_start = '../../static/models/clustering/kmeans'
    html_path_start = 'models/clustering/kmeans/labels'
    
    # datasets
    datasets = ['resorts', 'weather', 'google']
    
    # silhouette plots
    sil_cluster_plots = {dataset:[] for dataset in datasets}
    for cluster in range(2, 11):
        for dataset in datasets:
            sil_cluster_plots[dataset].append(os.path.join(img_path_start, f'{dataset}_clusters_{cluster}.png'))
            
    # average silhouette scores and elbow plots
    avg_sil_plots = {dataset: None for dataset in datasets}
    elbow_plots = {dataset: None for dataset in datasets}
    for dataset in datasets:
        avg_sil_plots[dataset] = os.path.join(img_path_start, f'coefficients_{dataset}.png')
        elbow_plots[dataset] = os.path.join(img_path_start, f'elbow_{dataset}.png')
        
    # optimal cluster choices
    resorts_clusters = [2, 3, 10]
    weather_clusters = [2, 3, 5]
    google_clusters = [4, 5, 6]
    
    # labels
    resorts_labels = ['Country', 'Pass', 'Region']
    weather_labels = ['icon', 'month']
    google_labels = ['Call Category', 'Country', 'Pass', 'Region']
    
    # datasets
    datasets = ['Resorts', 'Weather', 'Google']
    
    # pathway creation for cluster plotting
    plotting_pathways = {dataset:[] for dataset in datasets}
    plotting_titles = {dataset:[] for dataset in datasets}
    for dataset in datasets:
        if dataset=='Resorts':
            for cluster in resorts_clusters:
                for label in resorts_labels:
                    plotting_pathways[dataset].append(os.path.join(html_path_start, f'{dataset}_{label}_{cluster}.html'))
                    plotting_titles[dataset].append(f'Data: {dataset}<br>Label: {label}<br>Clusters: {cluster}<br>')
        elif dataset=='Weather':
            for cluster in weather_clusters:
                for label in weather_labels:
                    plotting_pathways[dataset].append(os.path.join(html_path_start, f'{dataset}_{label}_{cluster}.html'))
                    plotting_titles[dataset].append(f'Data: {dataset}<br>Label: {label}<br>Clusters: {cluster}<br>')
        else:
            for cluster in google_clusters:
                for label in google_labels:
                    plotting_pathways[dataset].append(os.path.join(html_path_start, f'{dataset}_{label}_{cluster}.html'))
                    plotting_titles[dataset].append(f'Data: {dataset}<br>Label: {label}<br>Clusters: {cluster}<br>')
    
    # returns
    return sil_cluster_plots, avg_sil_plots, elbow_plots, plotting_pathways, plotting_titles

# function to aid in hc clustering output
def formulate_clustering_output_hc():
    # base path
    img_path_start = '../../static/models/clustering/hierarchical'
    
    # files
    image_files = {'dendr_3_0': 'dendrogram_resorts_3_0.png',
                   'dendr_3_30': 'dendrogram_resorts_3_30.png',
                   'dendr_4_20': 'dendrogram_resorts_4_20.png',
                   'dendr_full': 'dendrogram_resorts_full.png',
                   'spread_3_0_country': 'spread_resorts_3_0_country.png',
                   'spread_3_0_pass': 'spread_resorts_3_0_pass.png',
                   'spread_3_0_region': 'spread_resorts_3_0_region.png',
                   'spread_3_30_country': 'spread_resorts_3_30_country.png',
                   'spread_3_30_pass': 'spread_resorts_3_30_pass.png',
                   'spread_3_30_region': 'spread_resorts_3_30_region.png',
                   'spread_4_20_country': 'spread_resorts_4_20_country.png',
                   'spread_4_20_pass': 'spread_resorts_4_20_pass.png',
                   'spread_4_20_region': 'spread_resorts_4_20_region.png'}
    
    # create paths
    image_pathways = {image_file: None for image_file in image_files.keys()}
    for image_file in image_files:
        image_pathways[image_file] = (os.path.join(img_path_start, image_files[image_file]))
    
    return image_pathways

# function to aid in dbscan output
def formulate_clustering_output_density():
    # base path
    img_path_start = '../../static/models/clustering/dbscan'
    # html path
    html_path_start = 'models/clustering/dbscan'
    
    # epsilon choices
    datasets = ['Resorts', 'Weather', 'Google']
    epsilon_choices = {dataset:[] for dataset in datasets}
    dbscan_clusters = {dataset:[] for dataset in datasets}
    dbscan_titles = {dataset:[] for dataset in datasets}
    
    for dataset in datasets:
        # auto epsilon
        epsilon_choices[dataset].append(os.path.join(img_path_start, f'auto_eps_{dataset}.png'))
        # manual epsilon
        epsilon_choices[dataset].append(os.path.join(img_path_start, f'manual_eps_{dataset}.png'))
        # auto epsilon html
        dbscan_clusters[dataset].append(os.path.join(html_path_start, f'{dataset.lower()}_auto.html'))
        dbscan_titles[dataset].append(f'{dataset} DBSCAN with Automatic Parameters')
        # manual epsilon html
        dbscan_clusters[dataset].append(os.path.join(html_path_start, f'{dataset.lower()}_manual.html'))
        dbscan_titles[dataset].append(f'{dataset} DBSCAN with Manual Parameters')
        # custom epsilon html
        if dataset=='Google':
            dbscan_clusters[dataset].append(os.path.join(html_path_start, f'{dataset.lower()}_custom.html'))
            dbscan_titles[dataset].append(f'{dataset} DBSCAN with Custom Parameters')
        
    # return pathways
    return epsilon_choices, dbscan_clusters, dbscan_titles
    
    
# render clustering model page
@model_pages_controls.route('/model_pages/model_clustering.html', methods=['GET', 'POST'])
def load_clustering_page():
    # final resort data - snippet
    df = pd.read_csv('snowbound/data/cleaned_data/resorts_snippet.csv')
    resorts_final = df.to_html(classes='table table-striped', index=False)
    
    # resort pca snippet
    df = pd.read_csv('snowbound/static/models/clustering/projection_snippet_resorts.csv')
    resorts_pca_snippet = df.to_html(classes='table table-striped', index=False, justify='left')
    
    # weather final - snippet
    df = pd.read_csv('snowbound/data/cleaned_data/weather_snippet.csv')
    weather_final = df.to_html(classes='table table-striped', index=False)
    
    # weather pca snippet
    df = pd.read_csv('snowbound/static/models/clustering/projection_snippet_weather.csv')
    weather_pca_snippet = df.to_html(classes='table table-striped', index=False, justify='left')
    
    # google types final - snippet
    df = pd.read_csv('snowbound/data/cleaned_data/google_places_snippet.csv')
    google_final = df.to_html(classes='table table-striped', index=False)
    
    # google pca snippet
    df = pd.read_csv('snowbound/static/models/clustering/projection_snippet_google.csv')
    google_pca_snippet = df.to_html(classes='table table-striped', index=False, justify='left')
    
    # function calls
    sil_cluster_plots, avg_sil_plots, elbow_plots, plotting_pathways, plotting_titles = formulate_clustering_output_kmeans()
    pca_snippets = [resorts_pca_snippet, weather_pca_snippet, google_pca_snippet]
    image_pathways = formulate_clustering_output_hc()
    epsilon_choices, dbscan_clusters, dbscan_titles = formulate_clustering_output_density()
    
    return render_template('model_pages/model_clustering.html',
                           resorts_final=resorts_final,
                           weather_final=weather_final,
                           google_final=google_final,
                           sil_cluster_plots=sil_cluster_plots,
                           avg_sil_plots=avg_sil_plots,
                           elbow_plots=elbow_plots,
                           plotting_pathways=plotting_pathways,
                           plotting_titles=plotting_titles,
                           pca_snippets=pca_snippets,
                           image_pathways=image_pathways,
                           epsilon_choices=epsilon_choices,
                           dbscan_clusters=dbscan_clusters,
                           dbscan_titles=dbscan_titles)

# function to aid in arm output
def formulate_arm_output():
    ## INTRODUCTION ##
    # html path
    introduction_path = '../../static/models/arm/introduction'
    
    # movie transaction list
    df = pd.read_csv('snowbound/static/models/arm/introduction/movie_list.csv')
    movie_list = df.to_html(classes='table table-striped', index=False, header=False, justify='left')
    
    # movie frequencies image
    movie_freq_img = os.path.join(introduction_path, 'movie_frequencies.png')
    
    # movie frequent itemsets
    df = pd.read_csv('snowbound/static/models/arm/introduction/movie_frequent_itemsets.csv')
    movie_itemsets = df.to_html(classes='table table-striped', index=True, justify='left')
    
    # movie association rules
    df = pd.read_csv('snowbound/static/models/arm/introduction/movie_association_rules.csv')
    movie_rules = df.to_html(classes='table table-striped', index=True, justify='left')
    
    # movie association rules image
    movie_rules_img = os.path.join(introduction_path, 'movie_rules.png')
    
    # create return list
    introduction = [movie_list, movie_freq_img, movie_itemsets, movie_rules, movie_rules_img]
    
    
    ## DATA PREP ##
    # initial dataset - snippet
    df = pd.read_csv('snowbound/static/models/arm/data_prep/google_unpacked_initial.csv')
    df_initial = df.to_html(classes='table table-striped', index=False, justify='left')
    
    # google places cleaned - snippet
    df = pd.read_csv('snowbound/data/cleaned_data/google_places_snippet.csv')
    google_final = df.to_html(classes='table table-striped', index=False, justify='left')
    
    # google merged final - snippet
    df = pd.read_csv('snowbound/static/models/arm/data_prep/google_merged.csv')
    google_merged = df.to_html(classes='table table-striped', index=False, justify='left')
    
    # transaction data itself
    transaction = pd.DataFrame(df['types'])
    transaction_data = transaction.to_html(classes='table table-striped', index=False, header=False, justify='left')
    
    data_prep = [df_initial, google_final, google_merged, transaction_data]
    
    ## RESULTS ##
    # top support
    top_15_support = pd.read_csv('snowbound/static/models/arm/data_prep/top_15_support.csv')
    top_15_support = top_15_support.to_html(classes='table table-striped', index=True, justify='left')
    top_15_support_lifted = pd.read_csv('snowbound/static/models/arm/data_prep/top_15_support_lifted.csv')
    top_15_support_lifted = top_15_support_lifted.to_html(classes='table table-striped', index=True, justify='left')
    
    # top confidence
    top_15_confidence = pd.read_csv('snowbound/static/models/arm/data_prep/top_15_confidence.csv')
    top_15_confidence = top_15_confidence.to_html(classes='table table-striped', index=True, justify='left')
    top_15_confidence_lifted = pd.read_csv('snowbound/static/models/arm/data_prep/top_15_confidence_lifted.csv')
    top_15_confidence_lifted = top_15_confidence_lifted.to_html(classes='table table-striped', index=True, justify='left')
    
    # top lift
    top_15_lift = pd.read_csv('snowbound/static/models/arm/data_prep/top_15_lift.csv')
    top_15_lift = top_15_lift.to_html(classes='table table-striped', index=True, justify='left')
    
    top_15_results = [top_15_support, top_15_support_lifted, top_15_confidence, top_15_confidence_lifted, top_15_lift]
    
    ## TOP RULES NETWORKS AND TITLES ##
    # html path
    html_path_start = 'models/arm/network_results'
    html_top_files = ['top_15_support_lifted', 'top_15_confidence_lifted', 'top_15_lift']
    top_rules_titles = ['Top 15 Rules by Support with Positive Correlation', 'Top 15 Rules by Confidence with Positive Correlation', 'Top 15 Rules by Lift']
    top_rules_networks = [os.path.join(html_path_start, f'{file}.html') for file in html_top_files]
    
    ## CALL CATEGORY LABEL EXPANSION ##
    # data
    rules_calls_df = pd.read_csv('snowbound/static/models/arm/data_prep/rules_calls.csv')
    rules_calls = rules_calls_df.head(10).to_html(classes='table table-striped', index=False, justify='left')
    
    # network
    rules_call_network = os.path.join(html_path_start, 'label_call.html')
    
    return introduction, data_prep, top_15_results, top_rules_networks, top_rules_titles, rules_calls, rules_call_network

# render arm model page
@model_pages_controls.route('/model_pages/model_arm.html', methods=['GET', 'POST'])
def load_arm_page():
    # function calls
    introduction, data_prep, top_15_results, top_rules_networks, top_rules_titles, rules_calls, rules_call_network = formulate_arm_output()
    
    return render_template('model_pages/model_arm.html',
                           introduction=introduction,
                           data_prep=data_prep,
                           top_15_results=top_15_results,
                           top_rules_networks=top_rules_networks,
                           top_rules_titles=top_rules_titles,
                           rules_calls=rules_calls,
                           rules_call_network=rules_call_network)

# render dt model page
@model_pages_controls.route('/model_pages/model_dt.html', methods=['GET', 'POST'])
def load_dt_page():
    # sample data
    sample_data = {'age': [25, 34, 45, 52, 23],
                   'blood_pressure': [120, 140, 130, 150, 115],
                   'cholesterol': [180, 220, 200, 240, 170],
                   'disease': ['False', 'True', 'False', 'True', 'False']}

    df_sample = pd.DataFrame(sample_data)
    sample_table = df_sample.to_html(classes='table table-striped', index=False, justify='left')
    
    # final resort data - snippet
    df = pd.read_csv('snowbound/data/cleaned_data/resorts_snippet.csv')
    resorts_final = df.to_html(classes='table table-striped', index=False, justify='left')
    
    # google places cleaned - snippet
    df = pd.read_csv('snowbound/data/cleaned_data/google_places_snippet.csv')
    google_final = df.to_html(classes='table table-striped', index=False, justify='left')
    
    # weather final - snippet
    df = pd.read_csv('snowbound/data/cleaned_data/weather_snippet.csv')
    weather_final = df.to_html(classes='table table-striped', index=False, justify='left')
    
    # decision tree prepared - snippet
    df = pd.read_csv('snowbound/static/models/dt/dt_prepared_snippet.csv')
    dt_prepared = df.to_html(classes='table table-striped', index=False, justify='left')
    
    # decision tree prepared train - snippet
    df = pd.read_csv('snowbound/static/models/dt/dt_default_train_snippet.csv')
    dt_prepared_train = df.to_html(classes='table table-striped', index=True, justify='left')
    
    # decision tree prepared test - snippet
    df = pd.read_csv('snowbound/static/models/dt/dt_default_test_snippet.csv')
    dt_prepared_test = df.to_html(classes='table table-striped', index=True, justify='left')
    
    # predicted label percents
    df = pd.read_csv('snowbound/static/models/dt/predicted_label_percents.csv')
    predictions = df.to_html(classes='table table-striped', index=False, justify='left')
    
    return render_template('model_pages/model_dt.html',
                           sample_table=sample_table,
                           resorts_final=resorts_final,
                           google_final=google_final,
                           weather_final=weather_final,
                           dt_prepared=dt_prepared,
                           dt_prepared_train=dt_prepared_train,
                           dt_prepared_test=dt_prepared_test,
                           predictions=predictions)

# render nb model page
@model_pages_controls.route('/model_pages/model_nb.html', methods=['GET', 'POST'])
def load_nb_page():
    img_path_start = 'snowbound/static/models/nb'
    
    # final resort data - snippet
    df = pd.read_csv('snowbound/data/cleaned_data/resorts_snippet.csv')
    resorts_final = df.to_html(classes='table table-striped', index=False, justify='left')
    
    # google places cleaned - snippet
    df = pd.read_csv('snowbound/data/cleaned_data/google_places_snippet.csv')
    google_final = df.to_html(classes='table table-striped', index=False, justify='left')
    
    # weather final - snippet
    df = pd.read_csv('snowbound/data/cleaned_data/weather_snippet.csv')
    weather_final = df.to_html(classes='table table-striped', index=False, justify='left')
    
    # multinomial data
    # multinomial main snippets
    df = pd.read_csv('snowbound/static/models/nb/multinomial_snippet.csv')
    multinomial_snippet = df.to_html(classes='table table-striped', index=False, justify='left')
    # multinomial train snippet
    df = pd.read_csv('snowbound/static/models/nb/multinomial_train_snippet.csv')
    multinomial_train_snippet = df.to_html(classes='table table-striped', index=False, justify='left')
    # multinomial test snippet
    df = pd.read_csv('snowbound/static/models/nb/multinomial_test_snippet.csv')
    multinomial_test_snippet = df.to_html(classes='table table-striped', index=False, justify='left')
    
    # gaussian data
    # gaussian main snippets
    df = pd.read_csv('snowbound/static/models/nb/gaussian_snippet.csv')
    gaussian_snippet = df.to_html(classes='table table-striped', index=False, justify='left')
    # gaussian train snippet
    df = pd.read_csv('snowbound/static/models/nb/gaussian_train_snippet.csv')
    gaussian_train_snippet = df.to_html(classes='table table-striped', index=False, justify='left')
    # gaussian test snippet
    df = pd.read_csv('snowbound/static/models/nb/gaussian_test_snippet.csv')
    gaussian_test_snippet = df.to_html(classes='table table-striped', index=False, justify='left')
    
    # bernoulli data
    # bernoulli main snippets
    df = pd.read_csv('snowbound/static/models/nb/bernoulli_snippet.csv')
    bernoulli_snippet = df.to_html(classes='table table-striped', index=False, justify='left')
    # bernoulli train snippet
    df = pd.read_csv('snowbound/static/models/nb/bernoulli_train_snippet.csv')
    bernoulli_train_snippet = df.to_html(classes='table table-striped', index=False, justify='left')
    # bernoulli test snippet
    df = pd.read_csv('snowbound/static/models/nb/bernoulli_test_snippet.csv')
    bernoulli_test_snippet = df.to_html(classes='table table-striped', index=False, justify='left')
    
    return render_template('model_pages/model_nb.html',
                           resorts_final=resorts_final,
                           google_final=google_final,
                           weather_final=weather_final,
                           multinomial_snippet=multinomial_snippet,
                           multinomial_train_snippet=multinomial_train_snippet,
                           multinomial_test_snippet=multinomial_test_snippet,
                           gaussian_snippet=gaussian_snippet,
                           gaussian_train_snippet=gaussian_train_snippet,
                           gaussian_test_snippet=gaussian_test_snippet,
                           bernoulli_snippet=bernoulli_snippet,
                           bernoulli_train_snippet=bernoulli_train_snippet,
                           bernoulli_test_snippet=bernoulli_test_snippet)

# render svm model page
@model_pages_controls.route('/model_pages/model_svm.html', methods=['GET', 'POST'])
def load_svm_page():
    return render_template('model_pages/model_svm.html')

# render regression model page
@model_pages_controls.route('/model_pages/model_regression.html', methods=['GET', 'POST'])
def load_regression_page():
    # final resort data - snippet
    df = pd.read_csv('snowbound/data/cleaned_data/resorts_snippet.csv')
    resorts_final = df.to_html(classes='table table-striped', index=False, justify='left')
    
    # regression snippet
    df = pd.read_csv('snowbound/static/models/regression/regression_snippet.csv')
    regression_snippet = df.to_html(classes='table table-striped', index=False, justify='left')
    
    # regression train snippet
    df = pd.read_csv('snowbound/static/models/regression/train_snippet.csv')
    train_snippet = df.to_html(classes='table table-striped', index=False, justify='left')
    
    # regression test snippet
    df = pd.read_csv('snowbound/static/models/regression/test_snippet.csv')
    test_snippet = df.to_html(classes='table table-striped', index=False, justify='left')
    
    return render_template('model_pages/model_regression.html',
                           resorts_final=resorts_final,
                           regression_snippet=regression_snippet,
                           train_snippet=train_snippet,
                           test_snippet=test_snippet)





