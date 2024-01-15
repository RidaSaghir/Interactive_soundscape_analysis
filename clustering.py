import pandas as pd
import plotly.offline as pyo
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram


class ClusteringVisualizer:
    def __init__(self, data):
        self.data = data
        self.scaled_data = []
        self.optimal_clusters = None

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




    def kmeans_clustering(self, columns, clusters_ideal, num_clusters):
        # Load data from the CSV file
        #data = pd.read_csv(csv_file)

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

        print(optimal_clusters)

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

        return fig

    def hierar_clustering(self):
        # Apply hierarchical clustering
        agg_cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
        hierarchical_labels = agg_cluster.fit_predict(self.scaled_data)

        # Create linkage matrix
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