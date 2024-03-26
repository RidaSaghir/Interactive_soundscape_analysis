import pandas as pd
import plotly.express as px
import json
import os

path_data = None
last_dataset = None
path_exp = None
config = None
clustering_rep = None
clustering_mode = None
dim_red_mode = None
clustering_filter = None
acoustic_region = None
def load_config():
    global path_data, last_dataset, path_exp, config, clustering_rep, clustering_mode, dim_red_mode, clustering_filter, acoustic_region
    config = json.load(open('config.json'))
    path_data = config["PATH_DATA"]
    last_dataset = config["last_dataset"]
    path_exp = os.path.join(os.path.dirname(path_data), 'exp')
    clustering_rep = config["clustering_rep"]
    clustering_mode = config["clustering_mode"]
    dim_red_mode = config["dim_red_mode"]
    clustering_filter = config["clustering_filter"]
    acoustic_region = config["acoustic_region"]

def cluster_occurrence_bar(which_cluster, cluster_x_axis, cluster_hue_b):
    load_config()
    csv = f'{clustering_rep}_{clustering_mode}_{dim_red_mode}_{clustering_filter}_{acoustic_region}.csv'
    result_file_path = os.path.join(os.path.dirname(path_data), "exp", last_dataset, csv)
    cluster_title = f'{clustering_mode} labels'
    df = pd.read_csv(result_file_path, index_col=0)
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