import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from feature_engine.selection import MRMR
import os


def pca_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    n_components: int = None,
    scree_plot: bool = False,
    save: bool = False,
    dataset_suffix: str = ""
):
    """
    Apply PCA to X_train and X_test, save transformed datasets with n_components.
    If n_components is not specified, use the minimum number of components that explain at least 80% of the variance.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        n_components (int, optional): Number of principal components to keep.
        scree_plot (bool, optional): Whether to plot the explained variance ratio.
        save (bool, optional): Whether to save the PCA-transformed data.
        dataset_suffix (str, optional): Suffix for saved dataset filenames.

    Returns:
        tuple: (X_train_pca, X_test_pca, explained_var)
    """
    # Use only numerical columns
    pca_features = [col for col in X_train.columns if X_train[col].dtype in [np.int64, np.float64, bool]]
    X_train_num = X_train[pca_features].astype(float)
    X_test_num = X_test[pca_features].astype(float)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_num)
    X_test_scaled = scaler.transform(X_test_num)

    # Fit PCA with all components for scree plot and variance calculation
    pca_full = PCA(n_components=min(X_train_scaled.shape), random_state=12345)
    pca_full.fit(X_train_scaled)
    explained_var_full = pca_full.explained_variance_ratio_

    # Scree plot
    if scree_plot:
        cumulative_var = explained_var_full.cumsum()
        plt.figure(figsize=(8, 4))
        plt.plot(np.arange(1, len(cumulative_var) + 1), cumulative_var, marker='o')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Scree Plot: All Principal Components')
        plt.grid(True)
        plt.show()

    # If n_components is not specified, choose minimum number to explain >=80% variance
    if n_components is None:
        cumulative_var = explained_var_full.cumsum()
        n_components = np.argmax(cumulative_var >= 0.80) + 1
        print(f"Number of components selected: {n_components}")

    # Fit PCA with chosen n_components
    pca = PCA(n_components=n_components, random_state=12345)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    explained_var = pca.explained_variance_ratio_

    if save:
        os.makedirs("./data/processed/PCA", exist_ok=True)
        pd.DataFrame(X_train_pca, columns=[f'PC{i+1}' for i in range(X_train_pca.shape[1])]).to_csv(
            f"./data/processed/PCA/pca_X_train{dataset_suffix}.csv", index=False)
        pd.DataFrame(X_test_pca, columns=[f'PC{i+1}' for i in range(X_test_pca.shape[1])]).to_csv(
            f"./data/processed/PCA/pca_X_test{dataset_suffix}.csv", index=False)

    return X_train_pca, X_test_pca,


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
    X_train = pd.read_csv("./data/processed/original_X_train.csv")
    X_test = pd.read_csv("./data/processed/original_X_test.csv")
    X_train_pca, X_test_pca = pca_features(X_train, X_test, scree_plot=True, save=True)