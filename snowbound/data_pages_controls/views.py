from flask import render_template, redirect, url_for, request, Blueprint, flash, send_from_directory
import pandas as pd
import os

data_pages_controls = Blueprint('data_pages_controls', __name__)

'''
render pages: source
'''
# render general page
@data_pages_controls.route('/data_pages/general.html', methods=['GET', 'POST'])
def load_general_page():
    return render_template('data_pages/general.html')

# render ski_resorts page
@data_pages_controls.route('/data_pages/ski_resorts.html', methods=['GET', 'POST'])
def load_ski_resorts_page():
    return render_template('data_pages/ski_resorts.html')

# render weather page
@data_pages_controls.route('/data_pages/weather.html', methods=['GET', 'POST'])
def load_weather_page():
    return render_template('data_pages/weather.html')

# render google page
@data_pages_controls.route('/data_pages/google.html', methods=['GET', 'POST'])
def load_google_page():
    return render_template('data_pages/google.html')

'''
render pages: cleaning
'''
# render ski_resorts_cleaning page
@data_pages_controls.route('/data_pages/ski_resorts_cleaning.html', methods=['GET', 'POST'])
def ski_resorts_cleaning_page():
    # main ski resort data - snippet
    df = pd.read_csv('snowbound/data/initial_data/main_resort_snippet.csv')
    main_resort_initial = df.to_html(classes='table table-striped', index=False)
    # main ski resort data - nulls
    df = pd.read_csv('snowbound/data/initial_data/main_resort_null.csv')
    main_resort_null = df.to_html(classes='table table-striped', index=False)
    
    # epic data - snippet
    df = pd.read_csv('snowbound/data/initial_data/epic_resort_snippet.csv')
    epic_initial = df.to_html(classes='table table-striped', index=False)
    # epic data - nulls
    df = pd.read_csv('snowbound/data/initial_data/epic_resort_null.csv')
    epic_null = df.to_html(classes='table table-striped', index=False)
    
    # ikon data - snippet
    df = pd.read_csv('snowbound/data/initial_data/ikon_resort_snippet.csv')
    ikon_initial = df.to_html(classes='table table-striped', index=False)
    # ikon data - nulls
    df = pd.read_csv('snowbound/data/initial_data/ikon_resort_null.csv')
    ikon_null = df.to_html(classes='table table-striped', index=False)
    
    # resort proper - snippet
    df = pd.read_csv('snowbound/data/initial_data/resort_proper_snippet.csv')
    resort_proper_initial = df.to_html(classes='table table-striped', index=False)
    # resort proper - nulls
    df = pd.read_csv('snowbound/data/initial_data/resort_proper_null.csv')
    resort_proper_null = df.to_html(classes='table table-striped', index=False)
    
    # resort proper - snippet
    df = pd.read_csv('snowbound/data/initial_data/address_unpacked_snippet.csv')
    address_unpacked_initial = df.to_html(classes='table table-striped', index=False)
    # resort proper - nulls
    df = pd.read_csv('snowbound/data/initial_data/address_unpacked_null.csv')
    address_unpacked_null = df.to_html(classes='table table-striped', index=False)
    
    # us regions - snippet
    df = pd.read_csv('snowbound/data/initial_data/region_us_snippet.csv')
    region_us_initial = df.to_html(classes='table table-striped', index=False)
    # us regions- nulls
    df = pd.read_csv('snowbound/data/initial_data/region_us_null.csv')
    region_us_null = df.to_html(classes='table table-striped', index=False)
    
    # canada regions - snippet
    df = pd.read_csv('snowbound/data/initial_data/region_canada_snippet.csv')
    region_canada_initial = df.to_html(classes='table table-striped', index=False)
    # canada regions - nulls
    df = pd.read_csv('snowbound/data/initial_data/region_canada_null.csv')
    region_canada_null = df.to_html(classes='table table-striped', index=False)
    
    # final data - snippet
    df = pd.read_csv('snowbound/data/cleaned_data/resorts_snippet.csv')
    resorts_final = df.to_html(classes='table table-striped', index=False)
    # final data - snippet
    df = pd.read_csv('snowbound/data/cleaned_data/resorts_null.csv')
    resorts_null = df.to_html(classes='table table-striped', index=False)
    
    return render_template('data_pages/ski_resorts_cleaning.html',
                           main_resort_initial=main_resort_initial,
                           main_resort_null=main_resort_null,
                           epic_initial=epic_initial,
                           epic_null=epic_null,
                           ikon_initial=ikon_initial,
                           ikon_null=ikon_null,
                           resort_proper_initial=resort_proper_initial,
                           resort_proper_null=resort_proper_null,
                           address_unpacked_initial=address_unpacked_initial,
                           address_unpacked_null=address_unpacked_null,
                           region_us_initial=region_us_initial,
                           region_us_null=region_us_null,
                           region_canada_initial=region_canada_initial,
                           region_canada_null=region_canada_null,
                           resorts_final=resorts_final,
                           resorts_null=resorts_null)

# render google_cleaning page
@data_pages_controls.route('/data_pages/google_cleaning.html', methods=['GET', 'POST'])
def google_cleaning_page():
    # google data initial - snippet
    df = pd.read_csv('snowbound/data/initial_data/google_places_snippet.csv')
    google_places_initial = df.to_html(classes='table table-striped', index=False)
    # google data initial - nulls
    df = pd.read_csv('snowbound/data/initial_data/google_places_null.csv')
    google_places_null = df.to_html(classes='table table-striped', index=False)
    
    # google types initial - snippet
    df = pd.read_csv('snowbound/data/initial_data/google_types_snippet.csv')
    google_types_initial = df.to_html(classes='table table-striped', index=False)
    # google types initial - nulls
    df = pd.read_csv('snowbound/data/initial_data/google_types_null.csv')
    google_types_null = df.to_html(classes='table table-striped', index=False)
    
    # google problematic initial - snippet
    df = pd.read_csv('snowbound/data/initial_data/google_places_problematic_snippet.csv')
    google_problem = df.to_html(classes='table table-striped', index=False)
    
    # google types final - snippet
    df = pd.read_csv('snowbound/data/cleaned_data/google_places_snippet.csv')
    google_final = df.to_html(classes='table table-striped', index=False)
    # google types final - nulls
    df = pd.read_csv('snowbound/data/cleaned_data/google_places_null.csv')
    google_final_null = df.to_html(classes='table table-striped', index=False)
    
    return render_template('data_pages/google_cleaning.html',
                           google_places_initial=google_places_initial,
                           google_places_null=google_places_null,
                           google_types_initial=google_types_initial,
                           google_types_null=google_types_null,
                           google_problem=google_problem,
                           google_final=google_final,
                           google_final_null=google_final_null)

# render weather_cleaning page
@data_pages_controls.route('/data_pages/weather_cleaning.html', methods=['GET', 'POST'])
def weather_cleaning_page():
    # weather data initial - snippet
    df = pd.read_csv('snowbound/data/initial_data/weather_snippet.csv')
    weather_initial = df.to_html(classes='table table-striped', index=False)
    # weather initial - nulls
    df = pd.read_csv('snowbound/data/initial_data/weather_null.csv')
    weather_initial_null = df.to_html(classes='table table-striped', index=False)
    
    # stations data initial - snippet
    df = pd.read_csv('snowbound/data/initial_data/stations_snippet.csv')
    stations_initial = df.to_html(classes='table table-striped', index=False)
    # weather initial - nulls
    df = pd.read_csv('snowbound/data/initial_data/stations_null.csv')
    stations_null = df.to_html(classes='table table-striped', index=False)
    
    # weather final - snippet
    df = pd.read_csv('snowbound/data/cleaned_data/weather_snippet.csv')
    weather_final = df.to_html(classes='table table-striped', index=False)
    # weather final - nulls
    df = pd.read_csv('snowbound/data/cleaned_data/weather_null.csv')
    weather_final_null = df.to_html(classes='table table-striped', index=False)
    
    return render_template('data_pages/weather_cleaning.html',
                           weather_initial=weather_initial,
                           weather_initial_null=weather_initial_null,
                           stations_initial=stations_initial,
                           stations_null=stations_null,
                           weather_final=weather_final,
                           weather_final_null=weather_final_null)

'''
render pages: eda
'''
# render data exploration page
@data_pages_controls.route('/data_pages/data_exploration.html', methods=['GET', 'POST'])
def load_data_exploration():
    # load total_correlations
    df = pd.read_csv('snowbound/data/cleaned_data/total_correlation.csv')
    total_corr = df.to_html(classes='table table-striped', index=False)
    
    return render_template('data_pages/data_exploration.html',
                           total_corr=total_corr)
