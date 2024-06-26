import plotly.express as px
import pandas as pd
import ast
import calendar
import os
import json
from utils import load_config



def whole_year_plot(radio_x_axis, radio_groupby, y_var, resolution):
    _, path_data, last_dataset, path_exp, clustering_rep, resolution_feature, _, _, _, _ = load_config()
    csv_file_path = os.path.join(path_exp, last_dataset, f'{clustering_rep}_{resolution_feature}.csv')
    df = pd.read_csv(csv_file_path)
    df['Date'] = pd.to_datetime(df['Date Time'])

    # Access individual details
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Day'] = df['Date'].dt.day
    df['Hour'] = df['Date'].dt.hour
    df['Minute'] = df['Date'].dt.minute
    df['Second'] = df['Date'].dt.second

    if radio_x_axis == 'year cycle':

        match resolution:
            case 'Monthly':
                df['Numeric Time'] = df['Month']
            case 'Daily':
                df['Numeric Time'] = df['Month'] + df['Day']/30
            case 'Hourly':
                df['Numeric Time'] = df['Month'] + df['Day']/30 + df['Hour']/720
            case 'Minutes':
                df['Numeric Time'] = df['Month'] + df['Day'] / 30 + df['Hour'] / 720 + df['Minute']/43200

        fig = px.scatter(df, x=df['Numeric Time'], y=y_var, color=radio_groupby, opacity=0.8, trendline='lowess',
                         hover_data={'Year': True, 'Month': True, 'Day': True, 'Hour': True})
        fig.update_layout(title='{} over Time with {} resolution'.format(y_var,resolution), xaxis_title='Months', yaxis_title='Average {} Value'.format(y_var))
        # Set custom x-axis ticks for months
        fig.update_xaxes(
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=calendar.month_abbr[1:13],
            range = [0, 13]
        )

    elif radio_x_axis == 'diel cycle':
        match resolution:
            case 'Hourly':
                df['Numeric Time'] = df['Hour']
            case 'Minutes':
                df['Numeric Time'] = df['Hour'] + df['Minute'] / 60

        fig = px.scatter(df, x=df['Numeric Time'], y=y_var, color=radio_groupby, opacity=0.5, trendline='lowess',
                         hover_data={'Year': True, 'Month': True, 'Day': True, 'Hour': True})
        fig.update_layout(title='{} over Time with {} resolution'.format(y_var,resolution), xaxis_title='Hours',
                          yaxis_title='Average {} Value'.format(y_var))

        fig.update_xaxes(
            tickmode='array',
            tickvals=df['Hour'],
            range=[0, 24]
        )


    elif radio_x_axis == 'linear':
        resolution_mapping = lambda resolution: 'Y' if resolution == 'Yearly' else 'M' if resolution == 'Monthly' else 'W' if resolution == 'Weekly' else 'D' if resolution == 'Daily' else 'H' if resolution == 'Hourly' else 'T' if resolution == 'Minutes' else None
        resolution_x = resolution_mapping(resolution)
        df.set_index('Date', inplace=True)
        df_filtered = df[[y_var, 'Year', 'Month', 'Week', 'Day', 'Hour', 'Minute']]
        df_resampled = df_filtered.resample(resolution_x).median().reset_index()
        df_resampled.dropna(inplace=True)
        fig = px.scatter(df_resampled, x='Date', y=y_var, color=radio_groupby, opacity=0.8, trendline='lowess',
                         hover_data={'Year': True, 'Month': True, 'Week': True, 'Day': True, 'Hour': True})
        fig.update_layout(title='{} over Time with {} resolution'.format(y_var,resolution), xaxis_title='Linear', yaxis_title='Average {} Value'.format(y_var))

    aci_whole = 'plotly_app.png'
    fig.write_image(aci_whole, width=1200, height=600)


    return fig




