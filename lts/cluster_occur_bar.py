import pandas as pd
import plotly.express as px
import json
import os

config = json.load(open('config.json'))
PATH_DATA = config["PATH_DATA"]
last_dataset = config["last_dataset"]
PATH_EXP = os.path.join(os.path.dirname(PATH_DATA), 'exp')

def cluster_occurrence_bar(which_cluster, cluster_x_axis, cluster_hue_b, which_cluster_result_bar):
    csv = f'{config["clustering_rep"]}_{config["clustering_mode"]}_{config["dim_red_mode"]}_{config["clustering_filter"]}_{config["acoustic_region"]}.csv'
    result_file_path = os.path.join(os.path.dirname(PATH_DATA), "exp", last_dataset, csv)
    cluster_title = f'{config["clustering_mode"]} labels'
    df = pd.read_csv(result_file_path)
    which_cluster = int(which_cluster.split()[1])
    cluster_df = df[df[cluster_title] == int(which_cluster)]
    cluster_df['Date'] = pd.to_datetime(df['Date'])

    cluster_df['Year'] = cluster_df['Date'].dt.year
    cluster_df['Month'] = cluster_df['Date'].dt.month
    cluster_df['Day'] = cluster_df['Date'].dt.day
    cluster_df['Hour'] = cluster_df['Date'].dt.hour

    first_column = None
    sec_column = None

    if cluster_hue_b == 'year':
        #first_column = cluster_df['Date'].dt.year
        first_column = 'Year'
    elif cluster_hue_b == 'month':
        #first_column = cluster_df['Date'].dt.month
        first_column = 'Month'
    elif cluster_hue_b == 'day':
        #first_column = cluster_df['Date'].dt.day
        first_column = 'Day'
    elif cluster_hue_b == 'hour':
        #first_column = cluster_df['Date'].dt.hour
        first_column = 'Hour'

    if cluster_x_axis == 'Year cycle':
        #sec_column = cluster_df['Date'].dt.month
        sec_column = 'Month'
    elif cluster_x_axis == 'Diel cycle':
        #sec_column = cluster_df['Date'].dt.hour
        sec_column = 'Hour'


    if first_column and sec_column and first_column != sec_column:
        df_count = cluster_df.groupby([first_column, sec_column]).size().reset_index(name='Count')
    elif first_column and sec_column and first_column == sec_column:
        df_count = cluster_df.groupby([first_column]).size().reset_index(name='Count')


    if cluster_x_axis == 'Linear cycle':
        cluster_df.set_index('Date', inplace=True)
        df_count = cluster_df.groupby(cluster_df.index.date).size().reset_index(name='Count')
        df_count.set_index('Date', inplace=True)
        fig = px.bar(df_count, x=df_count.index, y='Count', color=df_count.index.year,
                         labels={'Count of cluster occurrences': f'Occurrences of Cluster {which_cluster}'},
                         title=f'Occurrences of Cluster {which_cluster} ')
        return fig

    fig = px.bar(df_count, x=sec_column, y='Count', color=first_column,
                 labels=f'Occurrences of {cluster_df["assigned_labels"].iloc[0]}',
                 title=f'Occurrences of Cluster {cluster_df["assigned_labels"].iloc[0]} ')

    return fig