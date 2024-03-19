import pandas as pd
import plotly.express as px
import os
import json

config = json.load(open('config.json'))
PATH_DATA = config["PATH_DATA"]
last_dataset = config["last_dataset"]
PATH_EXP = os.path.join(os.path.dirname(PATH_DATA), 'exp')

def cluster_occurrence_rose(which_cluster_r, cluster_hue_r):
    csv = f'{config["clustering_rep"]}_{config["clustering_mode"]}_{config["dim_red_mode"]}_{config["clustering_filter"]}_{config["acoustic_region"]}.csv'
    result_file_path = os.path.join(os.path.dirname(PATH_DATA), "exp", last_dataset, csv)
    cluster_title = f'{config["clustering_mode"]} labels'
    df = pd.read_csv(result_file_path, index_col=0)
    which_cluster_r = int(which_cluster_r.split()[1])
    cluster_df = df[df[cluster_title] == int(which_cluster_r)]

    cluster_df['Date'] = pd.to_datetime(cluster_df['Date'])
    cluster_df['Year'] = cluster_df['Date'].dt.year
    cluster_df['Month'] = cluster_df['Date'].dt.month
    cluster_df['Day'] = cluster_df['Date'].dt.day
    cluster_df['Hour'] = cluster_df['Date'].dt.hour

    df_count = cluster_df.groupby([cluster_hue_r, 'Hour']).size().reset_index(name='Count')
    df_count = df_count.assign(r=(df_count["Hour"] / 24) * 360)
    fig = px.bar_polar(df_count, r="Count", theta="r",
                       color=cluster_hue_r, hover_data={"Hour"}, template="plotly_dark",
                       color_discrete_sequence= px.colors.sequential.Plasma_r)

    labelevery = 6
    fig.update_layout(
        polar={
            "angularaxis": {
                "tickmode": "array",
                "tickvals": list(range(0, 360, 360 // labelevery)),
                "ticktext": [f"{a:02}:00" for a in range(0, 24, 24 // labelevery)],
            }
        },
        title = f'Cluster {cluster_df["assigned_labels"].iloc[0]} occurrences'
    )
    return fig

