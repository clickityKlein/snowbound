from flask_wtf import FlaskForm
from wtforms import SubmitField, SelectField

class ScaleForm(FlaskForm):
    time_scale = SelectField('Temporal Aggregation', choices=[('month', 'Month'), ('week', 'Week')], default='month')
    spatial_scale = SelectField('Spatial Aggregation', choices=[('country', 'Country'), ('region', 'Region'), ('state', 'State and Province'), ('resort', 'Resort')], default='country')
    submit = SubmitField('Change Aggregation Scale')