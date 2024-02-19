import pandas as pd
import plotly.express as px
import os
import json

config = json.load(open('config.json',))
PATH_DATA = config["PATH_DATA"]
last_dataset = config["last_dataset"]
PATH_EXP = os.path.join(os.path.dirname(PATH_DATA), 'exp')

def cluster_occurrence_rose(which_cluster_r, cluster_hue_r, which_cluster_result_rose):
    if which_cluster_result_rose == 'pca':
        csv_file_path = os.path.join(os.path.dirname(PATH_DATA), "exp", last_dataset, "clustered_indices_pca.csv")
    elif which_cluster_result_rose == 'acoustic':
        csv_file_path = os.path.join(os.path.dirname(PATH_DATA), "exp", last_dataset, "clustered_indices_pca.csv")

    data_clustered = pd.read_csv(csv_file_path)
    cluster_df = data_clustered[data_clustered['KMeans_Cluster'] == int(which_cluster_r)]
    cluster_df = cluster_df.groupby([cluster_hue_r, 'Hour']).size().reset_index(name='Count')
    cluster_df = cluster_df.assign(r=(cluster_df["Hour"] / 24) * 360)
    fig = px.bar_polar(cluster_df, r="Count", theta="r",
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
        }
    )
    return fig

