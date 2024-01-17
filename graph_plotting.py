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
            fig = px.bar(df, x="Region", y="Value", color="Year_Month_Day", barmode="group",
                         labels={"Value": "Value", "Region": "Region"},
                         title="ACI values according to regions for different years")

            # Customize the layout
            fig.update_layout(
                xaxis_title="Region",
                yaxis_title="Value",
                xaxis_tickangle=45,
                bargap=0.2,
                showlegend=True,
                legend_title="Date",
                height=600,
                width=1200
            )
            # Plot using Seaborn
            # sns.set(style="whitegrid")
            # plt.figure(figsize=(12, 6))
            # ax = sns.barplot(data=df, x="Region", y="Value", hue="Year_Month_Day")
            # ax.set(xlabel="Region", ylabel="Value")
            # plt.xticks(rotation=45)
            # plt.title("ACI values according to regions for different dates")
            # # Display or save the plot
            # plt.grid(True)
            # region_wise = ('region_wise.png')
            # plt.savefig(region_wise, dpi=150)

        elif hue == 'Years on x-axis':
            # Plot 2: Years on x-axis, regions as hue
            fig = px.bar(df, x="Year_Month_Day", y="Value", color="Region", barmode="group",
                         labels={"Value": "Value", "Region": "Region"},
                         title="ACI values according to regions for different years")

            # Customize the layout
            fig.update_layout(
                xaxis_title="Years",
                yaxis_title="Value",
                xaxis_tickangle=45,
                bargap=0.2,
                showlegend=True,
                legend_title="Regions",
                height=600,
                width=1200
            )
            # plt.figure(figsize=(10, 6))
            # sns.barplot(data=df, x='Year_Month_Day', y='Value', hue='Region')
            # plt.xlabel('Years')
            # plt.ylabel('Average ACI Values')
            # plt.title('ACI Values by Region and Year')
            # plt.legend(title='Region')
            # # Display or save the plot
            # plt.grid(True)
            # region_wise = ('region_wise.png')
            # plt.savefig(region_wise, dpi=150)
            # plt.show()

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
            fig = px.line(df, x="Region", y="Value", color="Year_Month_Day",
                         labels={"Value": "Value", "Region": "Region"},
                         title="ACI values according to regions for different years")

            # Customize the layout
            fig.update_layout(
                xaxis_title="Regions",
                yaxis_title="Value",
                xaxis_tickangle=45,
                bargap=0.2,
                showlegend=True,
                legend_title="Date",
                height=600,
                width=1200
            )
            # plt.figure(figsize=(10, 6))
            # sns.lineplot(data=df, x='Region', y='Value', hue='Year_Month_Day')
            # plt.xlabel('Regions')
            # plt.ylabel('Average ACI Values')
            # plt.title('ACI Values by Region and Year')
            # plt.legend(title='Year')
            # # Display or save the plot
            # plt.grid(True)
            # region_wise = ('region_wise.png')
            # plt.savefig(region_wise, dpi=150)
            # plt.show()

        elif hue == 'Years on x-axis':
            # Plot 2: Years on x-axis, regions as hue
            fig = px.line(df, x="Year_Month_Day", y="Value", color="Region",
                         labels={"Value": "Value", "Region": "Region"},
                         title="ACI values according to regions for different years")

            # Customize the layout
            fig.update_layout(
                xaxis_title="Years",
                yaxis_title="Value",
                xaxis_tickangle=45,
                bargap=0.2,
                showlegend=True,
                legend_title="Regions",
                height=600,
                width=1200
            )
            # plt.figure(figsize=(10, 6))
            # sns.lineplot(data=df, x='Year_Month_Day', y='Value', hue='Region')
            # plt.xlabel('Years')
            # plt.ylabel('Average ACI Values')
            # plt.title('ACI Values by Region and Year')
            # plt.legend(title='Region')
            # # Display or save the plot
            # plt.grid(True)
            # region_wise = ('region_wise.png')
            # plt.savefig(region_wise, dpi=150)
            # plt.show()

    return fig


def aci_whole_plot(df, radio_x_axis, radio_groupby, y_var, resolution):

    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']], format = '%Y/%m/%d %H:%M')

    if radio_x_axis == 'year cycle':
        fig, ax = plt.subplots(figsize=(12, 6))

        # resolution_mapping = {
        #     'Monthly': lambda x: x['Date'].dt.month,
        #     'Weekly': lambda x: x['Date'].dt.month + x['Date'].dt.isocalendar().week / 4,
        #     'Daily': lambda x: x['Date'].dt.month + x['Date'].dt.isocalendar().week / 4 + x['Date'].dt.day / 31,
        #     'Hourly': lambda x: x['Date'].dt.month + x['Date'].dt.isocalendar().week / 4 + x['Date'].dt.day / 31 + x['Date'].dt.hour / 744,
        #     'Minutes': lambda x: x['Date'].dt.month + x['Date'].dt.isocalendar().week / 4 + x['Date'].dt.day / 31 + x['Date'].dt.hour / 744 + x[
        #         'Date'].dt.minute / 44640
        # }
        resolution_mapping = {
            'Monthly': 'M',
            'Weekly': 'W',
            'Daily': 'D',
            'Hourly': 'H',
            'Minutes': 'T'
        }

        if radio_groupby == 'Year':
            frequency = resolution_mapping.get(resolution)
            df.set_index('Date', inplace=True)
            df_filtered = df[[y_var, 'Year', 'Month', 'Week', 'Day', 'Hour', 'Minute']]
            df_resampled = df_filtered.resample(frequency).mean()
            fig = px.scatter(df_resampled, x=df_resampled.index.month, y=y_var, color='Year', opacity=0.8, trendline='lowess',
                             hover_data={'Year': True, 'Month': True, 'Day': True, 'Hour': True})
            fig.update_layout(title='{} over Time with {} resolution'.format(y_var,resolution), xaxis_title='Months', yaxis_title='Average {} Value'.format(y_var))
            # Set custom x-axis ticks for months
            fig.update_xaxes(
                tickmode='array',
                tickvals=list(range(1, 13)),
                ticktext=calendar.month_abbr[1:13],
                range = [0, 13]
            )
            aci_whole = 'plotly_app.png'
            fig.write_image(aci_whole, width=1200, height=600)
            # sns.set(style="whitegrid")
            # sns.scatterplot(x=df_resampled.index.month, y=y_var, hue='Year', data=df_resampled, palette='viridis', alpha=0.5, ax=ax)
            # plt.title('{} over Time with {} resolution'.format(y_var, resolution))
            # plt.xlabel('Months')
            # plt.ylabel('Average ACI Value')
            # plt.savefig('abc.png')
            # plt.show()


        if radio_groupby == 'Month':

            frequency = resolution_mapping.get(resolution)
            df.set_index('Date', inplace=True)
            df_filtered = df[[y_var, 'Year', 'Month', 'Week', 'Day', 'Hour', 'Minute']]
            df_resampled = df_filtered.resample(frequency).mean()

            fig = px.scatter(df_resampled, x=df_resampled.index.month, y=y_var, color='Month', opacity=0.8, trendline='lowess',
                             hover_data={'Year': True, 'Month': True, 'Day': True, 'Hour': True})
            fig.update_layout(title='{} over Time with {} resolution'.format(y_var,resolution), xaxis_title='Months', yaxis_title='Average {} Value'.format(y_var))
            # Set custom x-axis ticks for months
            fig.update_xaxes(
                tickmode='array',
                tickvals=list(range(1, 13)),
                ticktext=calendar.month_abbr[1:13],
                range = [0, 13]
            )
            aci_whole = 'plotly_app.png'
            fig.write_image(aci_whole, width=1200, height=600)
            # sns.set(style="whitegrid")
            # sns.scatterplot(x=df_resampled.index.month, y=y_var, hue='Month', data=df_resampled, palette='viridis', alpha=0.8, ax=ax)
            # plt.title('{} over Time with {} resolution'.format(y_var, resolution))
            # plt.xlabel('Months')
            # plt.ylabel('Average ACI Value')

        if radio_groupby == 'Week':

            frequency = resolution_mapping.get(resolution)
            df.set_index('Date', inplace=True)
            df_filtered = df[[y_var, 'Year', 'Month', 'Week', 'Day', 'Hour', 'Minute']]
            df_resampled = df_filtered.resample(frequency).mean()
            fig = px.scatter(df_resampled, x=df_resampled.index.month, y=y_var, color='Week', opacity=0.8, trendline='lowess',
                             hover_data={'Year': True, 'Month': True, 'Day': True, 'Hour': True})
            fig.update_layout(title='{} over Time with {} resolution'.format(y_var,resolution), xaxis_title='Months', yaxis_title='Average {} Value'.format(y_var))
            # Set custom x-axis ticks for months
            fig.update_xaxes(
                tickmode='array',
                tickvals=list(range(1, 13)),
                ticktext=calendar.month_abbr[1:13],
                range = [0, 13]
            )
            aci_whole = 'plotly_app.png'
            fig.write_image(aci_whole, width=1200, height=600)
            # sns.set(style="whitegrid")
            # sns.scatterplot(x=df_resampled.index.month, y=y_var, hue='Week', data=df_resampled, palette='viridis', alpha=0.8, ax=ax)
            # plt.title('{} over Time with {} resolution'.format(y_var, resolution))
            # plt.xlabel('Months')
            # plt.ylabel('Average ACI Value')

        if radio_groupby == 'Day':

            frequency = resolution_mapping.get(resolution)
            df.set_index('Date', inplace=True)
            df_filtered = df[[y_var, 'Year', 'Month', 'Week', 'Day', 'Hour', 'Minute']]
            df_resampled = df_filtered.resample(frequency).mean()
            fig = px.scatter(df_resampled, x=df_resampled.index.month, y=y_var, color='Day', opacity=0.8, trendline='lowess',
                             hover_data={'Year': True, 'Month': True, 'Day': True, 'Hour': True})
            fig.update_layout(title='{} over Time with {} resolution'.format(y_var,resolution), xaxis_title='Months', yaxis_title='Average {} Value'.format(y_var))
            # Set custom x-axis ticks for months
            fig.update_xaxes(
                tickmode='array',
                tickvals=list(range(1, 13)),
                ticktext=calendar.month_abbr[1:13],
                range = [0, 13]
            )
            aci_whole = 'plotly_app.png'
            fig.write_image(aci_whole, width=1200, height=600)
            # sns.set(style="whitegrid")
            # sns.scatterplot(x=df_resampled.index.month, y=y_var, hue='Day', data=df_resampled, palette='viridis',
            #                 alpha=0.8, ax=ax)
            # plt.title('{} over Time with {} resolution'.format(y_var, resolution))
            # plt.xlabel('Months')
            # plt.ylabel('Average ACI Value')

    if radio_x_axis == 'diel cycle':
        resolution_mapping = {
            'Hourly': lambda x: x['Date'].dt.hour,
            'Minutes': lambda x: x['Date'].dt.hour + x['Date'].dt.minute / 60 #60 minutes in an hour
        }

        if radio_groupby == 'Year':
            df['Hour_x'] = resolution_mapping.get(resolution)(df)
            fig = px.scatter(df, x='Hour_x', y=y_var, color='Year', opacity=0.8, trendline='lowess',
                             hover_data={'Year': True, 'Month': True, 'Day': True, 'Hour': True, 'Hour_x': False})
            fig.update_layout(title='Average ACI over whole day', xaxis_title='Hours', yaxis_title='Average {} Value'.format(y_var))
            # Set custom x-axis ticks for hours
            fig.update_xaxes(
                tickmode='array',
                tickvals=list(range(0, 23)),
                range = [-1, 24]
            )
            aci_whole = 'plotly_app.png'
            fig.write_image(aci_whole, width=1200, height=600)

        if radio_groupby == 'Month':
            df['Hour_x'] = resolution_mapping.get(resolution)(df)
            fig = px.scatter(df, x='Hour_x', y=y_var, color='Month', opacity=0.8, trendline='lowess',
                             hover_data={'Year': True, 'Month': True, 'Day': True, 'Hour': True, 'Hour_x': False})
            fig.update_layout(title='Average ACI over whole day', xaxis_title='Hours', yaxis_title='Average {} Value'.format(y_var))
            # Set custom x-axis ticks for hours
            fig.update_xaxes(
                tickmode='array',
                tickvals=list(range(0, 23)),
                range = [-1, 24]
            )
            aci_whole = 'plotly_app.png'
            fig.write_image(aci_whole, width=1200, height=600)

        if radio_groupby == 'Week':
            df['Hour_x'] = resolution_mapping.get(resolution)(df)
            fig = px.scatter(df, x='Hour_x', y=y_var, color='Week', opacity=0.8, trendline='lowess',
                             hover_data={'Year': True, 'Month': True, 'Day': True, 'Hour': True, 'Hour_x': False})
            fig.update_layout(title='Average ACI over whole day', xaxis_title='Hours', yaxis_title='Average {} Value'.format(y_var))
            # Set custom x-axis ticks for hours
            fig.update_xaxes(
                tickmode='array',
                tickvals=list(range(0, 23)),
                range = [-1, 24]
            )
            aci_whole = 'plotly_app.png'
            fig.write_image(aci_whole, width=1200, height=600)


        if radio_groupby == 'Day':
            df['Hour_x'] = resolution_mapping.get(resolution)(df)
            fig = px.scatter(df, x='Hour_x', y=y_var, color='Day', opacity=0.8, trendline='lowess',
                             hover_data={'Year': True, 'Month': True, 'Day': True, 'Hour': True, 'Hour_x': False})
            fig.update_layout(title='Average ACI over whole day', xaxis_title='Hours', yaxis_title='Average {} Value'.format(y_var))
            # Set custom x-axis ticks for hours
            fig.update_xaxes(
                tickmode='array',
                tickvals=list(range(0, 23)),
                range = [-1, 24]
            )
            aci_whole = 'plotly_app.png'
            fig.write_image(aci_whole, width=1200, height=600)

    if radio_x_axis == 'linear':
        resolution_mapping = lambda resolution: 'Y' if resolution == 'Yearly' else 'M' if resolution == 'Monthly' else 'W' if resolution == 'Weekly' else 'D' if resolution == 'Daily' else 'H' if resolution == 'Hourly' else 'T' if resolution == 'Minutes' else None
        fig, ax = plt.subplots(figsize=(12, 6))

        if radio_groupby == 'Year':
            resolution_x = resolution_mapping(resolution)
            df.set_index('Date', inplace=True)
            df_filtered = df[[y_var, 'Year', 'Month', 'Week', 'Day', 'Hour', 'Minute']]
            df_resampled = df_filtered.resample(resolution_x).mean().reset_index()
            fig = px.scatter(df_resampled, x='Date', y=y_var, color='Year', opacity=0.8, trendline='lowess',
                             hover_data={'Year': True, 'Month': True, 'Week': True, 'Day': True, 'Hour': True})
            fig.update_layout(title='{} over Time with {} resolution'.format(y_var,resolution), xaxis_title='Linear', yaxis_title='Average {} Value'.format(y_var))
            aci_whole = 'plotly_app.png'
            fig.write_image(aci_whole, width=1200, height=600)
            # sns.scatterplot(x='Date', y=y_var, hue='Year', data=df_resampled, palette='viridis', alpha=0.8, ax=ax)
            # plt.title('{} over Time with {} resolution'.format(y_var, resolution))
            # plt.xlabel('Linear')
            # plt.ylabel('Average ACI Value')


        if radio_groupby == 'Month':
            resolution_x = resolution_mapping(resolution)
            df.set_index('Date', inplace=True)
            df_filtered = df[[y_var, 'Year', 'Month', 'Week', 'Day', 'Hour', 'Minute']]
            df_resampled = df_filtered.resample(resolution_x).mean().reset_index()
            fig = px.scatter(df_resampled, x='Date', y=y_var, color='Month', opacity=0.8, trendline='lowess',hover_data={'Year': True, 'Month': True, 'Week': True, 'Day': True, 'Hour': True, 'Minute': True})
            fig.update_layout(title='{} over Time with {} resolution'.format(y_var,resolution), xaxis_title='Linear', yaxis_title='Average {} Value'.format(y_var))
            aci_whole = 'plotly_app.png'
            fig.write_image(aci_whole, width=1200, height=600)
            # sns.scatterplot(x='Date', y=y_var, hue='Month', data=df_resampled, palette='viridis', alpha=0.8, ax=ax)
            # plt.title('{} over Time with {} resolution'.format(y_var, resolution))
            # plt.xlabel('Linear')
            # plt.ylabel('Average ACI Value')

        if radio_groupby == 'Week':
            resolution_x = resolution_mapping(resolution)
            df.set_index('Date', inplace=True)
            df_filtered = df[[y_var, 'Year', 'Month', 'Week', 'Day', 'Hour', 'Minute']]
            df_resampled = df_filtered.resample(resolution_x).mean().reset_index()
            fig = px.scatter(df_resampled, x='Date', y=y_var, color='Week', opacity=0.8, trendline='lowess',hover_data={'Year': True, 'Month': True, 'Week': True, 'Day': True, 'Hour': True, 'Minute': True})
            fig.update_layout(title='{} over Time with {} resolution'.format(y_var,resolution), xaxis_title='Linear', yaxis_title='Average {} Value'.format(y_var))
            aci_whole = 'plotly_app.png'
            fig.write_image(aci_whole, width=1200, height=600)
            # sns.scatterplot(x='Date', y=y_var, hue='Week', data=df_resampled, palette='viridis', alpha=0.8, ax=ax)
            # plt.title('{} over Time with {} resolution'.format(y_var, resolution))
            # plt.xlabel('Linear')
            # plt.ylabel('Average ACI Value')



        if radio_groupby == 'Day':
            resolution_x = resolution_mapping(resolution)
            df.set_index('Date', inplace=True)
            df_filtered = df[[y_var, 'Year', 'Month', 'Week', 'Day', 'Hour', 'Minute']]
            df_resampled = df_filtered.resample(resolution_x).mean().reset_index()
            fig = px.scatter(df_resampled, x='Date', y=y_var, color='Day', opacity=0.8, trendline='lowess',hover_data={'Year': True, 'Month': True, 'Week': True, 'Day': True, 'Hour': True, 'Minute': True})
            fig.update_layout(title='{} over Time with {} resolution'.format(y_var,resolution), xaxis_title='Linear', yaxis_title='Average {} Value'.format(y_var))
            aci_whole = 'plotly_app.png'
            fig.write_image(aci_whole, width=1200, height=600)
            # sns.scatterplot(x='Date', y=y_var, hue='Day', data=df_resampled, palette='viridis', alpha=0.8, ax=ax)
            #
            # plt.title('{} over Time with {} resolution'.format(y_var, resolution))
            # plt.xlabel('Linear')
            # plt.ylabel('Average ACI Value')

    return fig