import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from sklearn.impute import SimpleImputer

def find_optimal_clusters(data, max_k=10):
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    # Initialize variables to store the best Silhouette score and the corresponding k
    best_score = -1
    best_k = 0

    # Iterate through different values of k
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k)
        data.loc[:, 'Cluster'] = kmeans.fit_predict(scaled_data)
        #data['Cluster'] = kmeans.fit_predict(scaled_data)
        silhouette_avg = silhouette_score(scaled_data, data['Cluster'])

        # Check if the current k has a better Silhouette score
        if silhouette_avg > best_score:
            best_score = silhouette_avg
            best_k = k

    return best_k



def kmeans_clustering(data, columns, clusters_ideal, num_clusters):
    # Load data from the CSV file
    #data = pd.read_csv(csv_file)

    # Select the columns for clustering
    selected_data = data[columns]

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(selected_data)

    optimal_clusters = 1
    if clusters_ideal == 'Get optimum number of clusters':
        optimal_clusters = find_optimal_clusters(selected_data)
    elif clusters_ideal == 'Choose random number of clusters':
        optimal_clusters = num_clusters

    print(optimal_clusters)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=optimal_clusters)
    data['Cluster'] = kmeans.fit_predict(scaled_data)

    # Create a dictionary to store cluster members (file names)
    cluster_members = {i: [] for i in range(optimal_clusters)}

    # Populate the cluster members dictionary
    for idx, row in data.iterrows():
        cluster_label = row['Cluster']
        file_name = row['File']
        cluster_members[cluster_label].append(file_name)

    # Plot the clusters
    plt.figure(figsize=(8, 6))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple', 'pink', 'lime', 'brown', 'gray', 'indigo']
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'H', 'X', '*', '+']

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for cluster_id in range(optimal_clusters):
        cluster_data = data[data['Cluster'] == cluster_id]
        ax.scatter(
            cluster_data[columns[0]],
            cluster_data[columns[1]],
            cluster_data[columns[2]],
            c=colors[cluster_id],
            marker=markers[cluster_id],
            label=f'Cluster {cluster_id}'
        )

    ax.set_xlabel(columns[0])
    ax.set_ylabel(columns[1])
    ax.set_zlabel(columns[2])
    ax.set_title('K-Means Clustering')
    ax.legend()
    plt.grid(True)

    clusters_pic = ('clusters.png')
    plt.savefig(clusters_pic, dpi=150)

    return clusters_pic

