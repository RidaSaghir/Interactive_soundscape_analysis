import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram


class ClusteringVisualizer:
    def __init__(self, data):
        self.data = data
        self.scaled_data = []
        self.optimal_clusters = None
        self.df_pca = []

    def find_optimal_clusters(self, data, max_k=10):
        # Standardize the data
        scaler = StandardScaler()
        self.scaled_data = scaler.fit_transform(data)
        # Initialize variables to store the best Silhouette score and the corresponding k
        best_score = -1
        best_k = 0

        # Iterate through different values of k
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k)
            data.loc[:, 'Cluster'] = kmeans.fit_predict(self.scaled_data)
            #data['Cluster'] = kmeans.fit_predict(scaled_data)
            silhouette_avg = silhouette_score(self.scaled_data, data['Cluster'])

            # Check if the current k has a better Silhouette score
            if silhouette_avg > best_score:
                best_score = silhouette_avg
                best_k = k

        return best_k

    def pca(self, num_dim, clusters_ideal, num_clusters):
        # Select only the columns with acoustic indices
        selected_data = self.data[['ACI', 'ENT', 'EVN', 'LFC', 'MFC', 'HFC', 'EPS']]  # Add other indices as needed

        # Standardize the data
        scaler = StandardScaler()
        self.scaled_data = scaler.fit_transform(selected_data)

        # Perform PCA
        pca = PCA(n_components=num_dim)  # Choose the number of components
        principal_components = pca.fit_transform(self.scaled_data)
        # Create a new DataFrame with the principal components
        columns = [f'Principal Component {i + 1}' for i in range(principal_components.shape[1])]
        self.df_pca = pd.DataFrame(data=principal_components, columns=columns)
        optimal_clusters = 1
        if clusters_ideal == 'Get optimum number of clusters':
            self.optimal_clusters = self.find_optimal_clusters(selected_data)
        elif clusters_ideal == 'Choose random number of clusters':
            self.optimal_clusters = num_clusters

        # Concatenate the original DataFrame with the new DataFrame
        result_df = pd.concat([self.data[['File', 'Timestamp']], self.df_pca], axis=1)
        result_df.to_csv('result_df.csv')
        return self.df_pca, selected_data, result_df, columns, self.optimal_clusters


    def kmeans_clustering(self, clustering, num_dim, columns, clusters_ideal, num_clusters):

        if clustering == 'pca':
            # This dict includes filename, timestamp, and PC columns
            self.df_pca, selected_data, result_df, columns, self.optimal_clusters = self.pca(num_dim, clusters_ideal, num_clusters)
            # Apply K-Means clustering
            kmeans = KMeans(n_clusters=self.optimal_clusters)
            self.data['KMeans_Cluster'] = kmeans.fit_predict(self.df_pca)
            # Create a dictionary to store cluster members (file names)
            cluster_members = {i: [] for i in range(self.optimal_clusters)}
            # Remove existing PCA columns from the DataFrame
            existing_pca_columns = [col for col in self.data.columns if col.startswith('Principal Component')]
            self.data = self.data.drop(columns=existing_pca_columns, errors='ignore')
            self.data.to_csv('self_data.csv')

            # Concatenate the new PCA columns
            self.data = pd.concat([self.data, self.df_pca], axis=1)


            # Populate the cluster members dictionary
            for idx, row in self.data.iterrows():
                cluster_label = row['KMeans_Cluster']
                file_name = row['File']
                cluster_members[cluster_label].append(file_name)
            #self.data = pd.concat([self.data, df_pca], axis = 1)

        if clustering == 'acoustic':
            # Select the columns for clustering
            selected_data = self.data[columns]

            # Standardize the data
            scaler = StandardScaler()
            self.scaled_data = scaler.fit_transform(selected_data)

            optimal_clusters = 1
            if clusters_ideal == 'Get optimum number of clusters':
                self.optimal_clusters = self.find_optimal_clusters(selected_data)
            elif clusters_ideal == 'Choose random number of clusters':
                self.optimal_clusters = num_clusters

            # Apply K-Means clustering
            kmeans = KMeans(n_clusters=self.optimal_clusters)
            self.data['KMeans_Cluster'] = kmeans.fit_predict(self.scaled_data)

            # Create a dictionary to store cluster members (file names)
            cluster_members = {i: [] for i in range(self.optimal_clusters)}

            # Populate the cluster members dictionary
            for idx, row in self.data.iterrows():
                cluster_label = row['KMeans_Cluster']
                file_name = row['File']
                cluster_members[cluster_label].append(file_name)

        # Plot the clusters
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple', 'pink', 'lime', 'brown', 'gray', 'indigo']
        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'H', 'X', '*', '+']

        fig = px.scatter_3d(
                self.data[self.data['KMeans_Cluster'] < self.optimal_clusters],  # Filter data for optimal clusters
                x=columns[0],
                y=columns[1],
                z=columns[2],
                color='KMeans_Cluster',
                labels={'Cluster': 'Cluster'},
                title='K-Means Clustering'
            )

        return fig, self.data

    def hierar_clustering(self, clustering):
        # Apply hierarchical clustering
        agg_cluster = AgglomerativeClustering(n_clusters=self.optimal_clusters, affinity='euclidean', linkage='ward')
        hierarchical_labels = agg_cluster.fit_predict(self.scaled_data)
        print(self.scaled_data)

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