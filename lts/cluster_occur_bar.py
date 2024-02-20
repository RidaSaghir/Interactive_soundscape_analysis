import pandas as pd
import plotly.express as px
import json
import os

config = json.load(open('config.json'))
PATH_DATA = config["PATH_DATA"]
last_dataset = config["last_dataset"]
PATH_EXP = os.path.join(os.path.dirname(PATH_DATA), 'exp')

def cluster_occurrence_bar(which_cluster, cluster_x_axis, cluster_hue_b, which_cluster_result_bar):
    if which_cluster_result_bar == 'pca':
        csv_file_path = os.path.join(os.path.dirname(PATH_DATA), "exp", last_dataset, "clustered_indices_pca.csv")
    elif which_cluster_result_bar == 'acoustic':
        csv_file_path = os.path.join(os.path.dirname(PATH_DATA), "exp", last_dataset, "clustered_indices_pca.csv")

    df = pd.read_csv(csv_file_path)
    which_cluster = int(which_cluster.split()[1])
    #Filtering the particular chosen cluster rows
    cluster_df = df[df['KMeans_Cluster'] == int(which_cluster)]
    cluster_df['Date'] = pd.to_datetime(df['Date'])

    cluster_df['Year'] = cluster_df['Date'].dt.year
    cluster_df['Month'] = cluster_df['Date'].dt.month
    cluster_df['Day'] = cluster_df['Date'].dt.day
    cluster_df['Hour'] = cluster_df['Date'].dt.hour


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


    if cluster_x_axis == 'Linear cycle':
        df_count = cluster_df.groupby('Date').size().reset_index(name='Count')
        fig = px.bar(df_count, x='Date', y='Count', color=first_column,
                         labels={'Count': f'Occurrences of Cluster {which_cluster}'},
                         title=f'Occurrences of Cluster {which_cluster} ')

    if first_column != sec_column:
        df_count = cluster_df.groupby([first_column, sec_column]).size().reset_index(name='Count')
    else:
        df_count = cluster_df.groupby([first_column]).size().reset_index(name='Count')

    fig = px.bar(df_count, x=sec_column, y='Count', color=first_column,
                         labels={'Count': f'Occurrences of Cluster {which_cluster}'},
                         title=f'Occurrences of Cluster {which_cluster} ')


    return fig