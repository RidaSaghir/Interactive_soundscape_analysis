import numpy as np
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast
from datetime import datetime
import calendar

def plot_aci_values_regions(df, plot, hue, region_type):

    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    group_region_averages = {}

    # Iterate through rows and accumulate values for each region within the same 'Year', 'Month', and 'Day' group
    for i, group in df.groupby(['Year']):
        total_rows_in_group = len(group)

        # Initialize the group's region averages that are present in region_type
        group_region_average = {region: 0 for region in region_type}

        # Iterate through rows within the group
        for index, row in group.iterrows():
            aci_values_str = row['ACI in regions']
            aci_values = ast.literal_eval(aci_values_str)
            print(aci_values)# Parse the string to a dictionary
            for region in region_type:
                group_region_average[region] += aci_values.get(region, 0)
                print(aci_values.get(region, 0))


        # Calculate the average for each region within the group
        for region in group_region_average:
            group_region_average[region] /= total_rows_in_group

        # Store the group's region averages in the dictionary
        group_region_averages[i] = group_region_average

    region_wise = None
    if plot == 'Bar plot':

        # Convert the dictionary to a list of dictionaries
        data = []
        for key, values in group_region_averages.items():
            data.append({'Year_Month_Day': "-".join(map(str, key)), **values})
        # Create a DataFrame
        df = pd.DataFrame(data)
        # Reshape the DataFrame
        df = df.melt(id_vars=['Year_Month_Day'], var_name='Region', value_name='Value')

        if hue == 'Regions on x-axis':
            # Plot 1: Regions on x-axis, years as hue

            # Plot using Seaborn
            sns.set(style="whitegrid")
            plt.figure(figsize=(12, 6))
            ax = sns.barplot(data=df, x="Region", y="Value", hue="Year_Month_Day")
            ax.set(xlabel="Region", ylabel="Value")
            plt.xticks(rotation=45)
            plt.title("ACI values according to regions for different dates")
            # Display or save the plot
            plt.grid(True)
            region_wise = ('region_wise.png')
            plt.savefig(region_wise, dpi=150)

        elif hue == 'Years on x-axis':
            # Plot 2: Years on x-axis, regions as hue
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df, x='Year_Month_Day', y='Value', hue='Region')
            plt.xlabel('Years')
            plt.ylabel('Average ACI Values')
            plt.title('ACI Values by Region and Year')
            plt.legend(title='Region')
            # Display or save the plot
            plt.grid(True)
            region_wise = ('region_wise.png')
            plt.savefig(region_wise, dpi=150)
            plt.show()

    elif plot == 'Time series plot':
        # Convert the dictionary to a list of dictionaries
        data = []
        for key, values in group_region_averages.items():
            data.append({'Year_Month_Day': "-".join(map(str, key)), **values})
        # Create a DataFrame
        df = pd.DataFrame(data)
        # Reshape the DataFrame
        df = df.melt(id_vars=['Year_Month_Day'], var_name='Region', value_name='Value')

        if hue == 'Regions on x-axis':
            # Plot 1: Regions on x-axis, years as hue
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=df, x='Region', y='Value', hue='Year_Month_Day')
            plt.xlabel('Regions')
            plt.ylabel('Average ACI Values')
            plt.title('ACI Values by Region and Year')
            plt.legend(title='Year')
            # Display or save the plot
            plt.grid(True)
            region_wise = ('region_wise.png')
            plt.savefig(region_wise, dpi=150)
            plt.show()

        elif hue == 'Years on x-axis':
            # Plot 2: Years on x-axis, years as hue
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=df, x='Year_Month_Day', y='Value', hue='Region')
            plt.xlabel('Years')
            plt.ylabel('Average ACI Values')
            plt.title('ACI Values by Region and Year')
            plt.legend(title='Region')
            # Display or save the plot
            plt.grid(True)
            region_wise = ('region_wise.png')
            plt.savefig(region_wise, dpi=150)
            plt.show()

    return region_wise


def aci_whole_plot(df, radio_x_axis, radio_groupby, y_variable, resolution):
    if y_variable == 'ACI (Acoustic Complexity Index)':
        y_var = 'ACI'
    elif y_variable == 'ENT (Temporal Entropy Index)':
        y_var = 'ENT'
    elif y_variable == 'CVR LF (Acoustic Cover Index - Low Freq)':
        y_var = 'LFC'
    elif y_variable == 'CVR MF (Acoustic Cover Index - Mid Freq)':
        y_var = 'MFC'
    elif y_variable == 'CVR HF (Acoustic Cover Index - High Freq)':
        y_var = 'HFC'
    elif y_variable == 'EVN (Event Count Index)':
        y_var = 'EVN'

    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']], format = '%Y/%m/%d %H:%M')

    if radio_x_axis == 'year cycle':
        resolution_mapping = {
            'Monthly': lambda x: x['Date'].dt.month,
            'Weekly': lambda x: x['Date'].dt.month + x['Date'].dt.isocalendar().week / 4,
            'Daily': lambda x: x['Date'].dt.month + x['Date'].dt.isocalendar().week / 4 + x['Date'].dt.day / 31,
            'Hourly': lambda x: x['Date'].dt.month + x['Date'].dt.isocalendar().week / 4 + x['Date'].dt.day / 31 + x['Date'].dt.hour / 744,
            'Minutes': lambda x: x['Date'].dt.month + x['Date'].dt.isocalendar().week / 4 + x['Date'].dt.day / 31 + x['Date'].dt.hour / 744 + x[
                'Date'].dt.minute / 44640
        }

        if radio_groupby == 'Year':
            df['Month_x'] = resolution_mapping.get(resolution)(df)
            sns.set(style="whitegrid")
            fig = px.scatter(df, x='Month_x', y=y_var, color='Year', opacity=0.8, trendline='lowess',
                             hover_data={'Year': True, 'Month': True, 'Day': True, 'Hour': True, 'Month_x': False})
            fig.update_layout(title='Average ACI over whole year', xaxis_title='Months', yaxis_title='Average ACI Value')
            # Set custom x-axis ticks for months
            fig.update_xaxes(
                tickmode='array',
                tickvals=list(range(1, 13)),
                ticktext=calendar.month_abbr[1:13],
                range = [0, 13]
            )
            aci_whole = 'plotly_app.png'
            fig.write_image(aci_whole, width=1200, height=600)

        if radio_groupby == 'Month':

            df['Month_x'] = resolution_mapping.get(resolution)(df)
            fig = px.scatter(df, x='Month_x', y=y_var, color='Month', opacity=0.8, trendline='lowess',
                             hover_data={'Year': True, 'Month': True, 'Day': True, 'Hour': True, 'Month_x': False})
            fig.update_layout(title='Average ACI over whole year', xaxis_title='Months', yaxis_title='Average ACI Value')
            # Set custom x-axis ticks for months
            fig.update_xaxes(
                tickmode='array',
                tickvals=list(range(1, 13)),
                ticktext=calendar.month_abbr[1:13],
                range = [0, 13]
            )
            aci_whole = 'plotly_app.png'
            fig.write_image(aci_whole, width=1200, height=600)

        if radio_groupby == 'Week':

            df['Month_x'] = resolution_mapping.get(resolution)(df)
            fig = px.scatter(df, x='Month_x', y=y_var, color='Week', opacity=0.8, trendline='lowess',
                             hover_data={'Year': True, 'Month': True, 'Day': True, 'Hour': True, 'Month_x': False})
            fig.update_layout(title='Average ACI over whole year', xaxis_title='Months', yaxis_title='Average ACI Value')
            # Set custom x-axis ticks for months
            fig.update_xaxes(
                tickmode='array',
                tickvals=list(range(1, 13)),
                ticktext=calendar.month_abbr[1:13],
                range = [0, 13]
            )
            aci_whole = 'plotly_app.png'
            fig.write_image(aci_whole, width=1200, height=600)

        if radio_groupby == 'Day':
            df['Month_x'] = resolution_mapping.get(resolution)(df)
            fig = px.scatter(df, x='Month_x', y=y_var, color='Day', opacity=0.8, trendline='lowess',
                             hover_data={'Year': True, 'Month': True, 'Day': True, 'Hour': True, 'Month_x': False})
            fig.update_layout(title='Average ACI over whole year', xaxis_title='Months', yaxis_title='Average ACI Value')
            # Set custom x-axis ticks for months
            fig.update_xaxes(
                tickmode='array',
                tickvals=list(range(1, 13)),
                ticktext=calendar.month_abbr[1:13],
                range = [0, 13]
            )
            aci_whole = 'plotly_app.png'
            fig.write_image(aci_whole, width=1200, height=600)

    if radio_x_axis == 'diel cycle':
        resolution_mapping = {
            'Hourly': lambda x: x['Date'].dt.hour,
            'Minutes': lambda x: x['Date'].dt.hour + x['Date'].dt.minute / 60 #60 minutes in an hour
        }

        if radio_groupby == 'Year':
            df['Hour_x'] = resolution_mapping.get(resolution)(df)
            fig = px.scatter(df, x='Hour_x', y=y_var, color='Year', opacity=0.8, trendline='lowess',
                             hover_data={'Year': True, 'Month': True, 'Day': True, 'Hour': True, 'Hour_x': False})
            fig.update_layout(title='Average ACI over whole day', xaxis_title='Hours', yaxis_title='Average ACI Value')
            # Set custom x-axis ticks for hours
            fig.update_xaxes(
                tickmode='array',
                tickvals=list(range(1, 25)),
                range = [1, 24]
            )
            aci_whole = 'plotly_app.png'
            fig.write_image(aci_whole, width=1200, height=600)

        if radio_groupby == 'Month':
            df['Hour_x'] = resolution_mapping.get(resolution)(df)
            fig = px.scatter(df, x='Hour_x', y=y_var, color='Month', opacity=0.8, trendline='lowess',
                             hover_data={'Year': True, 'Month': True, 'Day': True, 'Hour': True, 'Hour_x': False})
            fig.update_layout(title='Average ACI over whole day', xaxis_title='Hours', yaxis_title='Average ACI Value')
            # Set custom x-axis ticks for hours
            fig.update_xaxes(
                tickmode='array',
                tickvals=list(range(1, 25)),
                range = [1, 24]
            )
            aci_whole = 'plotly_app.png'
            fig.write_image(aci_whole, width=1200, height=600)

        if radio_groupby == 'Week':
            df['Hour_x'] = resolution_mapping.get(resolution)(df)
            fig = px.scatter(df, x='Hour_x', y=y_var, color='Week', opacity=0.8, trendline='lowess',
                             hover_data={'Year': True, 'Month': True, 'Day': True, 'Hour': True, 'Hour_x': False})
            fig.update_layout(title='Average ACI over whole day', xaxis_title='Hours', yaxis_title='Average ACI Value')
            # Set custom x-axis ticks for hours
            fig.update_xaxes(
                tickmode='array',
                tickvals=list(range(1, 25)),
                range = [1, 24]
            )
            aci_whole = 'plotly_app.png'
            fig.write_image(aci_whole, width=1200, height=600)


        if radio_groupby == 'Day':
            df['Hour_x'] = resolution_mapping.get(resolution)(df)
            fig = px.scatter(df, x='Hour_x', y=y_var, color='Day', opacity=0.8, trendline='lowess',
                             hover_data={'Year': True, 'Month': True, 'Day': True, 'Hour': True, 'Hour_x': False})
            fig.update_layout(title='Average ACI over whole day', xaxis_title='Hours', yaxis_title='Average ACI Value')
            # Set custom x-axis ticks for hours
            fig.update_xaxes(
                tickmode='array',
                tickvals=list(range(1, 25)),
                range = [1, 24]
            )
            aci_whole = 'plotly_app.png'
            fig.write_image(aci_whole, width=1200, height=600)

    if radio_x_axis == 'linear':
        resolution_mapping = lambda resolution: 'Y' if resolution == 'Yearly' else 'M' if resolution == 'Monthly' else 'W' if resolution == 'Weekly' else 'D' if resolution == 'Daily' else 'H' if resolution == 'Hourly' else 'T' if resolution == 'Minutes' else None

        if radio_groupby == 'Year':
            resolution_x = resolution_mapping(resolution)
            # Set 'Date' as the index
            df.set_index('Date', inplace=True)
            df_filtered = df[[y_var, 'Year']]
            df_resampled = df_filtered.resample(resolution_x).mean().reset_index()
            fig = px.line(df_resampled, x='Date', y='ACI', title='ACI over Time with {} resolution'.format(resolution))

            fig = px.scatter(df_resampled, x='Date', y=y_var, color='Year', opacity=0.8, trendline='lowess',
                             hover_data={'Date' : True})
            fig.update_layout(title='ACI over Time with {} resolution'.format(resolution), xaxis_title='Linear', yaxis_title='Average ACI Value')
            aci_whole = 'plotly_app.png'
            fig.write_image(aci_whole, width=1200, height=600)


        if radio_groupby == 'Month':
            df['Year_x'] = resolution_mapping.get(resolution)(df)
            fig = px.scatter(df, x='Year_x', y=y_var, color='Month', opacity=0.8, trendline='lowess',
                             hover_data={'Year': True, 'Month': True, 'Day': True, 'Hour': True, 'Year_x': False})
            fig.update_layout(title='Average ACI', xaxis_title='Linear', yaxis_title='Average ACI Value')
            aci_whole = 'plotly_app.png'
            fig.write_image(aci_whole, width=1200, height=600)

        if radio_groupby == 'Week':
            df['Year_x'] = resolution_mapping.get(resolution)(df)
            fig = px.scatter(df, x='Year_x', y=y_var, color='Week', opacity=0.8, trendline='lowess',
                             hover_data={'Year': True, 'Month': True, 'Day': True, 'Hour': True, 'Year_x': False})
            fig.update_layout(title='Average ACI', xaxis_title='Linear', yaxis_title='Average ACI Value')
            aci_whole = 'plotly_app.png'
            fig.write_image(aci_whole, width=1200, height=600)

        if radio_groupby == 'Day':
            df['Year_x'] = resolution_mapping.get(resolution)(df)
            fig = px.scatter(df, x='Year_x', y=y_var, color='Day', opacity=0.8, trendline='lowess',
                             hover_data={'Year': True, 'Month': True, 'Day': True, 'Hour': True, 'Year_x': False})
            fig.update_layout(title='Average ACI', xaxis_title='Linear', yaxis_title='Average ACI Value')
            aci_whole = 'plotly_app.png'
            fig.write_image(aci_whole, width=1200, height=600)

    return fig