import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
from pca import pca
import os
import json


config = json.load(open('config.json',))
PATH_DATA = config["PATH_DATA"]
last_dataset = config["last_dataset"]
PATH_EXP = os.path.join(os.path.dirname(PATH_DATA), 'exp')

csv_file_path = os.path.join(os.path.dirname(PATH_DATA), "exp", last_dataset, "all_indices.csv")
class ClusteringVisualizer:
    def __init__(self):
        self.data = pd.read_csv(csv_file_path)
        self.scaled_data = []
        self.optimal_clusters = None
        self.sil_score = None
        self.df_pca = []

    def find_optimal_clusters(self, data, max_clusters):
        max_k = int(max_clusters)

        # Initialize variables to store the best Silhouette score and the corresponding k
        best_score = -1
        best_k = 0
        silhouette_scores = []

        # Iterate through different values of k
        k_values = list(range(2, max_k + 1))  # Convert range object to a list

        for k in k_values:
            kmeans = KMeans(n_clusters=k, n_init=10)
            #data.loc[:, 'Cluster'] = kmeans.fit_predict(data)
            cluster_labels = kmeans.fit_predict(data)
            silhouette_avg = silhouette_score(data, cluster_labels)
            silhouette_scores.append(silhouette_avg)

            # Check if the current k has a better Silhouette score
            if silhouette_avg > best_score:
                best_score = silhouette_avg
                best_k = k
                self.optimal_clusters = k

        fig = px.line(x=k_values, y=silhouette_scores, markers=True,
                      labels={'x': 'Number of clusters (k)', 'y': 'Silhouette Score'},
                      title='Silhouette Score vs. Number of Clusters')

        return best_k, fig


    def kmeans_clustering(self, clustering, num_dim, columns, clusters_ideal, num_clusters, max_clusters):

        if clustering == 'pca':
            # Data already scaled in PCA function
            self.df_pca, result_df, columns = pca(self.data, num_dim)

            if clusters_ideal == 'Get optimum number of clusters':
                self.optimal_clusters, self.sil_score = self.find_optimal_clusters(self.df_pca, max_clusters)
            elif clusters_ideal == 'Choose the number of clusters':
                self.optimal_clusters = num_clusters

            # Apply K-Means clustering
            kmeans = KMeans(n_clusters=self.optimal_clusters)
            self.data['KMeans_Cluster'] = kmeans.fit_predict(self.df_pca)
            # Remove existing PCA columns from the DataFrame
            existing_pca_columns = [col for col in self.data.columns if col.startswith('Principal Component')]
            self.data = self.data.drop(columns=existing_pca_columns, errors='ignore')
            # Concatenate the new PCA columns (the whole original dataframe concatenated with PCs)
            self.data = pd.concat([self.data, self.df_pca], axis=1)


        if clustering == 'acoustic':
            # Select the columns for clustering
            selected_data = self.data[columns]
            # Standardize the data
            scaler = StandardScaler()
            self.scaled_data = scaler.fit_transform(selected_data)
            scaled_df = pd.DataFrame(self.scaled_data, columns=selected_data.columns)

            if clusters_ideal == 'Get optimum number of clusters':
                self.optimal_clusters, self.sil_score = self.find_optimal_clusters(scaled_df, max_clusters)
            elif clusters_ideal == 'Choose the number of clusters':
                self.optimal_clusters = num_clusters

            # Apply K-Means clustering
            kmeans = KMeans(n_clusters=self.optimal_clusters)
            print(self.scaled_data)
            self.data['KMeans_Cluster'] = kmeans.fit_predict(self.scaled_data)

        # Create a dictionary to store cluster members (file names)
        cluster_members = {i: [] for i in range(self.optimal_clusters)}
        # Populate the cluster members dictionary
        for idx, row in self.data.iterrows():
            cluster_label = row['KMeans_Cluster']
            file_name = row.index
            cluster_members[cluster_label].append(file_name)

        # Plot the clusters
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple', 'pink', 'lime', 'brown', 'gray', 'indigo']
        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'H', 'X', '*', '+']

        self.data.to_csv(os.path.join(os.path.dirname(PATH_DATA), "exp", last_dataset, "clustered_indices.csv"))
        print(PATH_DATA)
        print(last_dataset)
        fig = px.scatter_3d(
                self.data[self.data['KMeans_Cluster'] < self.optimal_clusters],  # Filter data for optimal clusters
                x=columns[0],
                y=columns[1],
                z=columns[2],
                color='KMeans_Cluster',
                labels={'Cluster': 'Cluster'},
                title='K-Means Clustering'
            )

        return fig, self.sil_score

    def hierar_clustering(self, clustering):
        # Apply hierarchical clustering
        agg_cluster = AgglomerativeClustering(n_clusters=self.optimal_clusters, affinity='euclidean', linkage='ward')
        hierarchical_labels = agg_cluster.fit_predict(self.scaled_data)

        if clustering == ('pca'):
            linkage_matrix = linkage(self.df_pca, method='ward')
        else:
            linkage_matrix = linkage(self.scaled_data, method='ward')

        # Create dendrogram
        dendrogram_fig, ax = plt.subplots(figsize=(12, 6))
        dendrogram(linkage_matrix, labels=self.data.index, leaf_rotation=90, leaf_font_size=10, truncate_mode='level',
                   p=5)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Samples')
        plt.ylabel('Distance')
        plt.tight_layout()

        # Save dendrogram as an image file (optional)
        dendrogram_image_path = 'hierarchical_dendrogram.png'
        plt.savefig(dendrogram_image_path)
        plt.close()

        return dendrogram_image_path