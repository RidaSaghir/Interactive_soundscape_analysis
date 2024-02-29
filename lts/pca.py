import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import json
import ast

config = json.load(open('config.json', ))
PATH_DATA = config["PATH_DATA"]
last_dataset = config["last_dataset"]
PATH_EXP = os.path.join(os.path.dirname(PATH_DATA), 'exp')


def pca(data_frame, num_dim, selected_columns):
    selected_data = data_frame[selected_columns]
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(selected_data)
    pca = PCA(n_components=num_dim)  # Choose the number of components
    principal_components = pca.fit_transform(scaled_data)
    # Create a new DataFrame with the principal components
    columns = [f'Principal Component {i + 1}' for i in range(principal_components.shape[1])]
    df_pca = pd.DataFrame(data=principal_components, columns=columns)

    # Concatenate the original DataFrame with the new DataFrame
    result_df = pd.concat([data_frame[['Date']], df_pca], axis=1)
    result_df.to_csv(os.path.join(os.path.dirname(PATH_DATA), "exp", last_dataset, "clustered_indices_pca.csv"))
    return df_pca, result_df, columns