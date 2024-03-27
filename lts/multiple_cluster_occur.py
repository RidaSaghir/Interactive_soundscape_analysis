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

def cluster_occurrence_multi(x_average):
    load_config()
    csv = f'{clustering_rep}_{clustering_mode}_{dim_red_mode}_{clustering_filter}_{acoustic_region}.csv'
    result_file_path = os.path.join(os.path.dirname(path_data), "exp", last_dataset, csv)
    cluster_title = f'{clustering_mode} labels'
    df = pd.read_csv(result_file_path, index_col=0)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    unique_clusters = sorted(df[cluster_title].unique())
    fig = px.line(title='Number of Cluster Occurrences over Time',
                  labels={'Date': 'Date', 'value': 'Occurrences', 'variable': 'Cluster'})
    df_count = pd.DataFrame()  # Initialize df_count with an empty DataFrame
    for cluster in unique_clusters:
        cluster_df = df[df[cluster_title] == cluster]
        if x_average == 'hour':
            df_count = cluster_df.groupby([cluster_title, cluster_df.index.hour]).size().reset_index(name='Count')
        elif x_average == 'date':
            df_count = cluster_df.groupby([cluster_title, cluster_df.index.date]).size().reset_index(name='Count')
        elif x_average == 'month':
            df_count = cluster_df.groupby([cluster_title, cluster_df.index.month]).size().reset_index(name='Count')

        if not df_count.empty:
            if 'assigned_labels' in df:
                title = df[df[cluster_title] == int(cluster)]['assigned_labels'].iloc[0]
                fig.add_scatter(x=df_count.iloc[:, 1], y=df_count['Count'], name=f'Cluster {title}')
            else:
                fig.add_scatter(x=df_count.iloc[:, 1], y=df_count['Count'], name=f'Cluster {cluster}')


    return fig

