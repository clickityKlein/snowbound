from flask import render_template, redirect, url_for, request, Blueprint, flash, send_from_directory
import pandas as pd

data_pages_controls = Blueprint('data_pages_controls', __name__)

# render general page
@data_pages_controls.route('/data_pages/general.html', methods=['GET', 'POST'])
def load_general_page():
    return render_template('data_pages/general.html')

# render ski_resorts page
@data_pages_controls.route('/data_pages/ski_resorts.html', methods=['GET', 'POST'])
def load_ski_resorts_page():
    # read in initial main header
    df = pd.read_csv('snowbound/data/initial_data/ski_resorts_main.csv')
    resort_main_initial = df.to_html(classes='table table-striped', index=False)
    
    return render_template('data_pages/ski_resorts.html', resort_main_initial=resort_main_initial)

# render weather page
@data_pages_controls.route('/data_pages/weather.html', methods=['GET', 'POST'])
def load_weather_page():
    return render_template('data_pages/weather.html')

# render google page
@data_pages_controls.route('/data_pages/google.html', methods=['GET', 'POST'])
def load_google_page():
    return render_template('data_pages/google.html')