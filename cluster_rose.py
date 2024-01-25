import pandas as pd
import plotly.express as px

def rose_plots(data_clustered, which_cluster_r, cluster_hue_r):

    cluster_df = data_clustered[data_clustered['KMeans_Cluster'] == int(which_cluster_r)]
    cluster_df = cluster_df.groupby([cluster_hue_r, 'Hour']).size().reset_index(name='Count')
    cluster_df = cluster_df.assign(r=(cluster_df["Hour"] / 24) * 360)
    print(cluster_df)
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

