import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from feature_engine.selection import MRMR
import os


def pca_features(data_encoded, n_components=None, scree_plot=False, save=False, dataset_suffix : str = ""):
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
        pca_df.to_csv(f"./data/processed/PCA/pca_{dataset_suffix}.csv", index=False)

    return X_pca, explained_var


def mrmr_features(data : pd.DataFrame, target: str, max_features: int = None, plotQ: bool = False) -> pd.DataFrame:
    """
    Select features using mRMR (Minimum Redundancy Maximum Relevance).
    
    Parameters:
    - data: DataFrame containing the dataset.
    - target: Name of the target variable.
    - max_features: Number of features to select.
    
    Returns:
    - DataFrame with selected features.
    """
    if max_features is None:
        max_features = len(data.columns) - 1
    if max_features <= 0:
        raise ValueError("max_features must be a positive integer.")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if not isinstance(target, str):
        raise TypeError("target must be a string representing the target variable name.")
    if target not in data.columns:
        raise ValueError(f"Target '{target}' not found in DataFrame columns.")
    if not isinstance(max_features, int):
        raise TypeError("max_features must be an integer.")
    if max_features > len(data.columns) - 1:
        raise ValueError("max_features cannot be greater than the number of features in the DataFrame.")
    

    # Ensure target is in the DataFrame
    if target not in data.columns:
        raise ValueError(f"Target '{target}' not found in DataFrame columns.")
    
    # Prepare features and target
    X = data.drop(columns=[target, 'ID'])
    y = data[target]
    
    # Instantiate MRMR selector
    mrmr_selector = MRMR(
        variables=None,  # Use all features
        max_features=max_features,
        scoring='auto',  # Default scoring method
        random_state=12345,
        discrete_features=X.select_dtypes(include=['category', 'bool']).columns.tolist(),
    )
    
    # Fit and transform the data
    X_mrmr = mrmr_selector.fit_transform(X, y)

    # Get selected feature names
    selected_features = X_mrmr.columns.tolist()
    print(f"Selected features by mRMR: {selected_features}")


    # Get scores and feature names
    scores = mrmr_selector.relevance_
    features = mrmr_selector.variables_

    if plotQ:
        # Plot relevance scores
        plt.figure(figsize=(15, 4))
        plt.bar(features, scores, color='skyblue')
        plt.xticks(rotation=90)
        plt.title("Relevance Scores of Features")
        plt.xlabel("Features")
        plt.ylabel("Relevance Score")
        plt.show()

    pd.Series(scores, index=features).sort_values(
        ascending=False).plot.bar(figsize=(15, 4))
    plt.title("Relevance")
    plt.show()
    
    return X_mrmr

    # Apply mRMR feature selection
    selected_data = mrmr_features(data_encoded, target='default payment next month', max_features=20)

if __name__ == "__main__":
    data = pd.read_csv("./data/processed/original_encoded.csv")
    print("Data loaded successfully. Shape:", data.shape, "Columns:", data.columns.tolist())
    print("Data types:", data.dtypes)
    # Apply mRMR feature selection
    selected_data = mrmr_features(data, target='default payment next month', max_features=20)
