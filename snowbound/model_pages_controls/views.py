from flask import render_template, redirect, url_for, request, Blueprint, flash, send_from_directory

model_pages_controls = Blueprint('model_pages_controls', __name__)

# render pca model page
@model_pages_controls.route('/model_pages/model_pca.html', methods=['GET', 'POST'])
def load_pca_page():
    return render_template('model_pages/model_pca.html')

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
