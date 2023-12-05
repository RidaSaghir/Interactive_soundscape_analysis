import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast
from datetime import datetime
import calendar

def plot_aci_values_regions(df, plot, hue, region_type):

    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    # Convert the 'dates' input to a datetime format for comparison
    #selected_dates = pd.to_datetime(dates, format='%Y/%m/%d')
    # Filter the DataFrame based on the selected dates
    #df = df[df['Date'].isin(selected_dates)]
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


def aci_whole_plot(df, radio_x_axis, radio_groupby):

    if radio_x_axis == 'year cycle' and radio_groupby == 'Year':

        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']], format = '%Y/%m/%d %H:%M')
        df['Month_x'] = df['Date'].dt.month + df['Date'].dt.day / 31 + df['Date'].dt.hour / 744 # Assuming 31 days in each month
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 6))
        #sns.lmplot(x='Date', y=averages.values, hue='Year', data=averages)
        sns.lmplot(x='Month_x', y='ACI', hue='Year', data=df, fit_reg=True, scatter_kws={'alpha':0.15})
        plt.xlabel('Months')
        plt.ylabel('Average ACI Value')
        plt.title('Average ACI over whole year')
        plt.xticks(range(1, 13), calendar.month_abbr[1:13])
        aci_whole = 'gradio_app.png'
        plt.savefig(aci_whole, dpi=150)
        plt.show()

    if radio_x_axis == 'year cycle' and radio_groupby == 'Month':

        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']], format = '%Y/%m/%d %H:%M')
        df['Month_x'] = df['Date'].dt.month + df['Date'].dt.day / 31 + df['Date'].dt.hour / 744 # Assuming 31 days in each month
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 6))
        #sns.lmplot(x='Date', y=averages.values, hue='Year', data=averages)
        sns.lmplot(x='Month_x', y='ACI', hue='Month', data=df, fit_reg=True, scatter_kws={'alpha':0.15})
        plt.xlabel('Months')
        plt.ylabel('Average ACI Value')
        plt.title('Average ACI over whole year')
        plt.xticks(range(1, 13), calendar.month_abbr[1:13])
        aci_whole = 'gradio_app.png'
        plt.savefig(aci_whole, dpi=150)
        plt.show()

    if radio_x_axis == 'year cycle' and radio_groupby == 'Week':

        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']], format = '%Y/%m/%d %H:%M')
        df['Month_x'] = df['Date'].dt.month + df['Date'].dt.day / 31  + df['Date'].dt.hour / 744 # Assuming 31 days in each month
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 6))
        sns.lmplot(x='Month_x', y='ACI', hue='Week', data=df, fit_reg=True, scatter_kws={'alpha':0.15})
        plt.xlabel('Months')
        plt.ylabel('Average ACI Value')
        plt.title('Average ACI over whole year')
        plt.xticks(range(1, 13), calendar.month_abbr[1:13])
        aci_whole = 'gradio_app.png'
        plt.savefig(aci_whole, dpi=150)
        plt.show()

    if radio_x_axis == 'year cycle' and radio_groupby == 'Day':
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']], format='%Y/%m/%d %H:%M')
        df['Month_x'] = df['Date'].dt.month + df['Date'].dt.day / 31  + df['Date'].dt.hour / 744 # Assuming 31 days in each month
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 6))
        # sns.lmplot(x='Date', y=averages.values, hue='Year', data=averages)
        sns.lmplot(x='Month_x', y='ACI', hue='Day', data=df, fit_reg=True, scatter_kws={'alpha':0.15})
        plt.xlabel('Months')
        plt.ylabel('Average ACI Value')
        plt.title('Average ACI over whole year')
        plt.xticks(range(1, 13), calendar.month_abbr[1:13])
        aci_whole = 'gradio_app.png'
        plt.savefig(aci_whole, dpi=150)
        plt.show()

    if radio_x_axis == 'diel cycle' and radio_groupby == 'Year':
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']], format='%Y/%m/%d %H:%M')
        df['Hour_x'] = df['Date'].dt.hour + df['Date'].dt.minute / 60  # Assuming 60 minutes in each hour
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 6))
        # sns.lmplot(x='Date', y=averages.values, hue='Year', data=averages)
        sns.lmplot(x='Hour_x', y='ACI', hue='Year', data=df, fit_reg=True, scatter_kws={'alpha':0.15}, lowess=True)
        plt.xlabel('Hours')
        plt.ylabel('Average ACI Value')
        plt.title('Average ACI over whole day')
        plt.xticks(range(1, 25))
        aci_whole = 'gradio_app.png'
        plt.savefig(aci_whole, dpi=150)
        plt.show()

    if radio_x_axis == 'diel cycle' and radio_groupby == 'Month':
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']], format='%Y/%m/%d %H:%M')
        df['Hour_x'] = df['Date'].dt.hour + df['Date'].dt.minute / 60  # Assuming 60 minutes in each hour
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 6))
        sns.lmplot(x='Hour_x', y='ACI', hue='Month', data=df, fit_reg=True, scatter_kws={'alpha':0.15}, lowess=True)
        plt.xlabel('Hours')
        plt.ylabel('Average ACI Value')
        plt.title('Average ACI over whole day')
        plt.xticks(range(1, 25))
        aci_whole = 'gradio_app.png'
        plt.savefig(aci_whole, dpi=150)
        plt.show()

    if radio_x_axis == 'diel cycle' and radio_groupby == 'Week':
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']], format='%Y/%m/%d %H:%M')
        df['Hour_x'] = df['Date'].dt.hour + df['Date'].dt.minute / 60  # Assuming 60 minutes in each hour
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 6))
        # sns.lmplot(x='Date', y=averages.values, hue='Year', data=averages)
        sns.lmplot(x='Hour_x', y='ACI', hue='Week', data=df, fit_reg=True, scatter_kws={'alpha':0.15}, lowess=True)
        plt.xlabel('Hours')
        plt.ylabel('Average ACI Value')
        plt.title('Average ACI over whole day')
        plt.xticks(range(1, 25))
        aci_whole = 'gradio_app.png'
        plt.savefig(aci_whole, dpi=150)
        plt.show()


    if radio_x_axis == 'diel cycle' and radio_groupby == 'Day':
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']], format='%Y/%m/%d %H:%M')
        df['Hour_x'] = df['Date'].dt.hour + df['Date'].dt.minute / 60  # Assuming 60 minutes in each hour
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 6))
        # sns.lmplot(x='Date', y=averages.values, hue='Year', data=averages)
        sns.lmplot(x='Hour_x', y='ACI', hue='Day', data=df, scatter_kws={'alpha':0.15}, lowess=True, fit_reg=True)
        plt.xlabel('Hours')
        plt.ylabel('Average ACI Value')
        plt.title('Average ACI over whole day')
        plt.xticks(range(1, 25))
        aci_whole = 'gradio_app.png'
        plt.savefig(aci_whole, dpi=150)
        plt.show()

    if radio_x_axis == 'linear' and radio_groupby == 'Year':
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']], format='%Y/%m/%d %H:%M')
        df[('Year_x')] = df['Year'] + df['Date'].dt.month / 12 + df['Date'].dt.day / 365  + df['Date'].dt.hour / 8760 # Assuming 31 days in each month
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 6))
        sns.lmplot(x='Year_x', y='ACI', hue='Year', data=df, fit_reg=True, scatter_kws={'alpha':0.15}, lowess=True)
        plt.xlabel('Months')
        plt.ylabel('Average ACI Value')
        plt.title('Average ACI over whole year')
        aci_whole = 'gradio_app.png'
        plt.savefig(aci_whole, dpi=150)
        plt.show()

    if radio_x_axis == 'linear' and radio_groupby == 'Month':
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']], format='%Y/%m/%d %H:%M')
        df[('Year_x')] = df['Year'] + df['Date'].dt.month / 12 + df['Date'].dt.day / 365  + df['Date'].dt.hour / 8760 # Assuming 31 days in each month
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 6))
        sns.lmplot(x='Year_x', y='ACI', hue='Month', data=df, fit_reg=True, scatter_kws={'alpha':0.15}, lowess=True)
        plt.xlabel('Months')
        plt.ylabel('Average ACI Value')
        plt.title('Average ACI over whole year')
        aci_whole = 'gradio_app.png'
        plt.savefig(aci_whole, dpi=150)
        plt.show()

    if radio_x_axis == 'linear' and radio_groupby == 'Week':
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']], format='%Y/%m/%d %H:%M')
        df[('Year_x')] = df['Year'] + df['Date'].dt.month / 12 + df['Date'].dt.day / 365 + df[
            'Date'].dt.hour / 8760  # Assuming 31 days in each month
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 6))
        sns.lmplot(x='Year_x', y='ACI', hue='Week', data=df, fit_reg=True, scatter_kws={'alpha': 0.15}, lowess=True)
        plt.xlabel('Months')
        plt.ylabel('Average ACI Value')
        plt.title('Average ACI over whole year')
        aci_whole = 'gradio_app.png'
        plt.savefig(aci_whole, dpi=150)
        plt.show()

    if radio_x_axis == 'linear' and radio_groupby == 'Day':
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']], format='%Y/%m/%d %H:%M')
        df[('Year_x')] = df['Year'] + df['Date'].dt.month / 12 + df['Date'].dt.day / 365 + df[
            'Date'].dt.hour / 8760  # Assuming 31 days in each month
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 6))
        sns.lmplot(x='Year_x', y='ACI', hue='Day', data=df, fit_reg=True, scatter_kws={'alpha': 0.15}, lowess=True)
        plt.xlabel('Months')
        plt.ylabel('Average ACI Value')
        plt.title('Average ACI over whole year')
        aci_whole = 'gradio_app.png'
        plt.savefig(aci_whole, dpi=150)
        plt.show()
    return aci_whole