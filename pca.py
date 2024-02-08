import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler





def pca(data_frame, num_dim, clusters_ideal, num_clusters, max_clusters):

    # Perform PCA
    selected_data = data_frame[
        ['ACI', 'ENT', 'EVN', 'LFC', 'MFC', 'HFC', 'EPS', 'EAS', 'ECV']]  # Add other indices as needed
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(selected_data)
    pca = PCA(n_components=num_dim)  # Choose the number of components
    principal_components = pca.fit_transform(scaled_data)
    # Create a new DataFrame with the principal components
    columns = [f'Principal Component {i + 1}' for i in range(principal_components.shape[1])]
    df_pca = pd.DataFrame(data=principal_components, columns=columns)

    # Concatenate the original DataFrame with the new DataFrame
    result_df = pd.concat([data_frame[['File', 'Timestamp']], df_pca], axis=1)
    result_df.to_csv('result_df.csv')
    return df_pca, result_df, columns