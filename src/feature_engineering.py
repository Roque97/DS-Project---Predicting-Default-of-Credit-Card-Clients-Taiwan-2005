import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



def pca_features(data_encoded, n_components = None, scree_plot = False):
    """
    Perform PCA on the provided encoded data and optionally plot the explained variance.
    
    Args:
        data_encoded (pd.DataFrame): The preprocessed and one-hot encoded data.
        n_components (int, optional): Number of principal components to keep. If None, all components are kept.
        scree_plot (bool, optional): Whether to plot the explained variance ratio.
        
    Returns:
        tuple: Transformed data and explained variance ratio.
    """
    assert isinstance(n_components, (int, type(None))), "n_components must be an integer or None"
    assert isinstance(data_encoded, pd.DataFrame), "data_encoded must be a pandas DataFrame"

    if n_components is None:
        n_components = min(data_encoded.shape[0], data_encoded.shape[1])
    
    # Prepare data for PCA: use only numerical columns from data_encoded
    # Exclude ID and target variable
    pca_features = [col for col in data_encoded.columns if data_encoded[col].dtype in [np.int64, np.float64, bool] and col not in ['default payment next month']]
    X = data_encoded[pca_features].astype(float)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Explained variance
    explained_var = pca.explained_variance_ratio_
    
    if scree_plot:
        cumulative_var = explained_var.cumsum()
        plt.figure(figsize=(8, 4))
        plt.plot(np.arange(1, len(cumulative_var) + 1), cumulative_var, marker='o')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA on Mixed Data (Numerical + Dummified Categorical)')
        plt.grid(True)
        plt.show()

    return X_pca, explained_var




