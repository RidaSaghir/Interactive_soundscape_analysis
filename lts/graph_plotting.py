import plotly.express as px
import pandas as pd
import ast
import calendar
import os
import json

config_file_path = os.path.join(os.path.dirname(__file__), 'config.json')
with open(config_file_path) as config_file:
    config = json.load(config_file)

path_data = config["PATH_DATA"]
last_dataset = config["last_dataset"]
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


    return fig


def whole_year_plot(dd_ds, radio_x_axis, radio_groupby, y_var, resolution):
    csv_file_path = os.path.join(os.path.dirname(path_data), "exp", last_dataset, "all_indices.csv")
    print(csv_file_path)
    df = pd.read_csv(csv_file_path)

    df['Date'] = pd.to_datetime(df['Date'])

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

        fig = px.scatter(df, x=df['Numeric Time'], y=y_var, color=radio_groupby, opacity=0.8, trendline='lowess',
                         hover_data={'Year': True, 'Month': True, 'Day': True, 'Hour': True})
        fig.update_layout(title='Average ACI over whole day', xaxis_title='Hours',
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
        if radio_groupby == 'KMeans_Cluster':
            df_filtered = df[[y_var, 'Year', 'Month', 'Week', 'Day', 'Hour', 'Minute', 'KMeans_Cluster']]
        else:
            df_filtered = df[[y_var, 'Year', 'Month', 'Week', 'Day', 'Hour', 'Minute']]
        df_resampled = df_filtered.resample(resolution_x).mean().reset_index()
        fig = px.scatter(df_resampled, x='Date', y=y_var, color=radio_groupby, opacity=0.8, trendline='lowess',
                         hover_data={'Year': True, 'Month': True, 'Week': True, 'Day': True, 'Hour': True})
        fig.update_layout(title='{} over Time with {} resolution'.format(y_var,resolution), xaxis_title='Linear', yaxis_title='Average {} Value'.format(y_var))

    aci_whole = 'plotly_app.png'
    fig.write_image(aci_whole, width=1200, height=600)


    return fig