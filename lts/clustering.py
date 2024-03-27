
import plotly.express as px
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, OPTICS, DBSCAN, HDBSCAN, SpectralClustering, AffinityPropagation
from sklearn.manifold import TSNE, Isomap
import os
import json
from acoustic_region_filter import region_filter
from acoustic_region_bp_filter import region_filter_bp
from sklearn.decomposition import PCA
import numpy as np
from utils import load_config




class ClusteringVisualizer:
    def __init__(self):
        self.optimal_clusters = None
        self.sil_score = None

    def load_config(self):
        (self.config, self.path_data, self.last_dataset, self.path_exp,
         self.clustering_rep, self.clustering_mode, self.dim_red_mode, self.clustering_filter, self.acoustic_region) = load_config()

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

    def scaler(self, df):
        scaler = StandardScaler()
        try:
            if np.isinf(df).any().any():
                df = df.replace([np.inf, -np.inf], np.nan)
                df = df.dropna(axis=1)  # Will drop the rows having nan values (data points of audio files)
        except Exception as e:
            print(f'{e}')
        scaled_data = scaler.fit_transform(df)
        self.scaled_df = pd.DataFrame(scaled_data)
        return self.scaled_df



    def clustering(self, acoustic_region, num_dim, indices, clusters_ideal, num_clusters, max_clusters):

        self.load_config()
        self.csv_file_path = os.path.join(self.path_exp, self.last_dataset,
                                          f'{self.clustering_rep}.csv')
        if self.csv_file_path and os.path.exists(self.csv_file_path):
            self.data = pd.read_csv(self.csv_file_path, index_col=0)
        else:
            self.data = None

        # if self.clustering_filter == 'acoustic region':
        #     # selected cols have per_bin indices (without date time info)
        #     self.data, selected_cols = region_filter(self.data, acoustic_region)
        #     scaled_df = self.scaler(self.data[selected_cols])
        if self.clustering_filter == 'acoustic region':
            self.data = region_filter_bp(self.data, acoustic_region)
            per_bin_columns = [col for col in self.data.columns if 'per_bin' in col]
            to_scale = self.data.drop(['Date'] + ['frequencies'] + ['LTS'] + per_bin_columns, axis=1).copy()
            scaled_df = self.scaler(to_scale).dropna(axis=1)
            scaled_df = scaled_df.set_index(self.data.index)
            self.data = pd.concat([self.data['Date'], scaled_df], axis=1)
        else:
            per_bin_columns = [col for col in self.data.columns if 'per_bin' in col]
            to_scale = self.data.drop(['Date'] + ['frequencies'] + ['LTS'] + per_bin_columns, axis=1).copy()
            scaled_df = self.scaler(to_scale).dropna(axis=1)
            scaled_df = scaled_df.set_index(self.data.index)
            self.data = pd.concat([self.data['Date'], scaled_df], axis=1)


        # Clustering Methods
        if self.clustering_mode == 'k_means':
            if clusters_ideal == 'Get optimum number of clusters':
                self.optimal_clusters, self.sil_score = self.find_optimal_clusters(scaled_df, max_clusters)
            elif clusters_ideal == 'Choose the number of clusters':
                self.optimal_clusters = num_clusters
            cluster_labels = KMeans(n_clusters=self.optimal_clusters, random_state=42, n_init='auto').fit_predict(scaled_df)
        elif self.clustering_mode == 'hdbscan':
            cluster_labels = HDBSCAN(n_jobs=-1).fit_predict(scaled_df)
        elif self.clustering_mode == 'dbscan':
            cluster_labels = DBSCAN(eps=0.001, min_samples=5, n_jobs=-1).fit_predict(scaled_df)

        # Dimensionality Reduction Methods
        if self.dim_red_mode == 'pca':
            dim_red_components = PCA(n_components=num_dim).fit_transform(scaled_df)
        elif self.dim_red_mode == 'tsne':
            dim_red_components = TSNE(n_components=num_dim, perplexity=min(len(scaled_df) // 2, 30), n_jobs=-1, learning_rate='auto', random_state=42).fit_transform(scaled_df)


        component_columns = [f'{self.dim_red_mode} component {i + 1}' for i in
                             range((dim_red_components).shape[1])]
        self.data[component_columns] = dim_red_components
        cluster_title = f'{self.clustering_mode} labels'
        self.data[cluster_title] = cluster_labels

        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple', 'pink', 'lime', 'brown', 'gray', 'indigo']
        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'H', 'X', '*', '+']

        results_csv = f'{self.clustering_rep}_{self.clustering_mode}_{self.dim_red_mode}_{self.clustering_filter}_{self.acoustic_region}.csv'
        self.data.to_csv(os.path.join(self.path_exp, self.last_dataset, results_csv))

        fig = px.scatter_3d(
                self.data,  # Filter data for optimal clusters
                x=component_columns[0],
                y=component_columns[1],
                z=component_columns[2],
                color=self.data[cluster_title],
                title='Clustering Results'
            )


        return fig, self.sil_score

