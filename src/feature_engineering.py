import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os



def pca_features(data_encoded, n_components=None, scree_plot=False, save=False):
    """
    Perform PCA on the provided encoded data and optionally plot the explained variance.

    Args:
        data_encoded (pd.DataFrame): The preprocessed and one-hot encoded data.
        n_components (int, optional): Number of principal components to keep. If None, all components are kept.
        scree_plot (bool, optional): Whether to plot the explained variance ratio.
        save (bool, optional): Whether to save the PCA-transformed data.

    Returns:
        tuple: Transformed data (with n_components) and explained variance ratio (for n_components).
    """
    assert isinstance(n_components, (int, type(None))), "n_components must be an integer or None"
    assert isinstance(data_encoded, pd.DataFrame), "data_encoded must be a pandas DataFrame"

    # Prepare data for PCA: use only numerical columns from data_encoded
    pca_features = [col for col in data_encoded.columns if data_encoded[col].dtype in [np.int64, np.float64, bool] and col not in ['default payment next month']]
    X = data_encoded[pca_features].astype(float)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit PCA with all components for scree plot
    pca_full = PCA(n_components=min(X_scaled.shape), random_state=12345)
    X_pca_full = pca_full.fit_transform(X_scaled)
    explained_var_full = pca_full.explained_variance_ratio_

    # Scree plot always shows all PCs
    if scree_plot:
        cumulative_var = explained_var_full.cumsum()
        plt.figure(figsize=(8, 4))
        plt.plot(np.arange(1, len(cumulative_var) + 1), cumulative_var, marker='o')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Scree Plot: All Principal Components')
        plt.grid(True)
        plt.show()

    # Now fit PCA with n_components for output and saving
    if n_components is None:
        n_components = min(X_scaled.shape)
    pca = PCA(n_components=n_components, random_state=12345)
    X_pca = pca.fit_transform(X_scaled)
    explained_var = pca.explained_variance_ratio_

    if save:
        os.makedirs("./data/processed/PCA", exist_ok=True)
        pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
        pca_df.to_csv(f"./data/processed/PCA/pca_transformed_{n_components}features.csv", index=False)

    return X_pca, explained_var

if __name__ == "__main__":
    data = pd.read_csv("./data/processed/original_X_train.csv")
    print("Data loaded successfully. Shape:", data.shape, "Columns:", data.columns.tolist())
    # To create the PCA features, we can call the function with the encoded data.
    pca_features(data, n_components=17, scree_plot=True, save=True)




