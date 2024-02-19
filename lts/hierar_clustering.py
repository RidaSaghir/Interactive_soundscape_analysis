import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
import json
import os
import pandas as pd
import sys

config = json.load(open('config.json',))
PATH_DATA = config["PATH_DATA"]
last_dataset = config["last_dataset"]
PATH_EXP = os.path.join(os.path.dirname(PATH_DATA), 'exp')

csv_file_path = os.path.join(os.path.dirname(PATH_DATA), "exp", last_dataset, "clustered_indices.csv")
def hierarchical_clustering(num_clusters_hierar, clustering_param_hierar, cluster_indices):
    data = pd.read_csv(csv_file_path)

    if clustering_param_hierar == 'acoustic':
        selected_data = data[cluster_indices]

    if clustering_param_hierar == 'pca':
        pca_columns = [col for col in data.columns if col.startswith('Principal Component')]
        selected_data = data[pca_columns]

    sys.setrecursionlimit(100000)
    # Ward Linkage
    linkage_data = linkage(selected_data, method='ward', metric='euclidean')
    dendrogram(linkage_data)
    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(linkage_data, ax=ax)
    ax.set_xlabel('Samples')
    ax.set_ylabel('Distance')
    ax.set_title('Dendrogram')
    dendrogram_image_path = 'hierarchical_dendrogram.png'
    plt.savefig(dendrogram_image_path)
    plt.close()


    # Hierarchical clusters
    hierarchical_cluster = AgglomerativeClustering(n_clusters=num_clusters_hierar, affinity='euclidean', linkage='ward')
    labels = hierarchical_cluster.fit_predict(selected_data)

    # Plot the clusters
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple', 'pink', 'lime', 'brown', 'gray', 'indigo']
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'H', 'X', '*', '+']

    hierar_clusters = px.scatter_3d(
        selected_data,
        x=selected_data.columns[0],
        y=selected_data.columns[1],
        z=selected_data.columns[2],
        color=labels,
        title='Hierarchical Clustering'
    )



    return dendrogram_image_path, hierar_clusters