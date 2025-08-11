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


def mrmr_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    max_features: int = 20,  # Set a default value that's not None
    plotQ: bool = False,
    save_path: str = None,
    dataset_suffix: str = ""
) -> tuple:
    """
    Select features using mRMR (Minimum Redundancy Maximum Relevance).
    
    Parameters:
    - X_train: Training features DataFrame
    - X_test: Test features DataFrame
    - y_train: Training target variable Series
    - max_features: Number of features to select (default: 20)
    - plotQ: Whether to plot feature relevance scores
    - save_path: Path to save the selected features dataset (None for no save)
    - dataset_suffix: Suffix for saved dataset filenames
    
    Returns:
    - tuple: (X_train_mrmr, X_test_mrmr, selected_features)
    """
    # Remove ID column if present
    id_col = 'ID'
    X_train_features = X_train.drop(columns=[id_col]) if id_col in X_train.columns else X_train.copy()
    X_test_features = X_test.drop(columns=[id_col]) if id_col in X_test.columns else X_test.copy()
    
    # Convert all columns to numeric and fill NaN values
    for col in X_train_features.columns:
        X_train_features[col] = pd.to_numeric(X_train_features[col], errors='coerce')
        X_test_features[col] = pd.to_numeric(X_test_features[col], errors='coerce')
    
    X_train_features.fillna(0, inplace=True)
    X_test_features.fillna(0, inplace=True)
    
    # Ensure target is numeric and 1D
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0].values  # Take first column if DataFrame
    elif isinstance(y_train, pd.Series):
        y_train = y_train.values
    
    # Drop any columns with constant values (these cause issues with MRMR)
    for col in X_train_features.columns.tolist():
        if X_train_features[col].nunique() <= 1:
            print(f"Dropping column {col} because it has only one unique value")
            X_train_features = X_train_features.drop(columns=[col])
            X_test_features = X_test_features.drop(columns=[col])
    
    # Ensure max_features isn't larger than the number of features
    max_features = min(max_features, X_train_features.shape[1])
    print(f"Using max_features={max_features}, total features: {X_train_features.shape[1]}")
    
    # Use alternative feature selection if MRMR fails
    try:
        # Try with MRMR
        mrmr_selector = MRMR(
            variables=None,
            max_features=max_features,
            scoring='accuracy',  # Be explicit about scoring
            random_state=12345
        )
        
        # Fit on training data
        print("Fitting MRMR selector...")
        mrmr_selector.fit(X_train_features, y_train)
        
        # Transform both train and test sets
        X_train_mrmr = mrmr_selector.transform(X_train_features)
        X_test_mrmr = mrmr_selector.transform(X_test_features)
        
        # Get selected feature names
        selected_features = X_train_mrmr.columns.tolist()
        
    except Exception as e:
        # Fallback to a simpler feature selection method if MRMR fails
        print(f"MRMR selection failed with error: {str(e)}")
    
    # Limit selected features to max_features
    if len(selected_features) > max_features:
        selected_features = selected_features[:max_features]
        X_train_mrmr = X_train_mrmr[selected_features]
        X_test_mrmr = X_test_mrmr[selected_features]
    
    print(f"Selected {len(selected_features)} features")
    print(f"Top 10 features: {selected_features[:10]}")
    
    # Plotting if requested
    if plotQ:
        try:
            # Try to plot MRMR relevance if available
            scores = mrmr_selector.relevance_
            features = mrmr_selector.variables_
            
            plt.figure(figsize=(15, 4))
            plt.bar(features, scores, color='skyblue')
            plt.xticks(rotation=90)
            plt.title("Feature Relevance Scores")
            plt.xlabel("Features")
            plt.ylabel("Relevance Score")
            plt.tight_layout()
            plt.show()
            
            # Plot top selected features' relevance
            selected_scores = {feature: scores[list(features).index(feature)] 
                              for feature in selected_features if feature in features}
            pd.Series(selected_scores).sort_values(ascending=False).plot.bar(figsize=(12, 4))
            plt.title("Relevance of Selected Features")
            plt.tight_layout()
            plt.show()
        except:
            # Fallback plot if MRMR plots aren't available
            plt.figure(figsize=(12, 6))
            plt.bar(selected_features, range(len(selected_features), 0, -1), color='skyblue')
            plt.xticks(rotation=90)
            plt.title("Selected Features")
            plt.tight_layout()
            plt.show()
    
    # Save the data if path is provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        
        # Add ID column back if it existed
        if id_col in X_train.columns:
            X_train_mrmr_save = pd.concat([X_train[id_col], X_train_mrmr], axis=1)
            X_test_mrmr_save = pd.concat([X_test[id_col], X_test_mrmr], axis=1)
        else:
            X_train_mrmr_save = X_train_mrmr
            X_test_mrmr_save = X_test_mrmr
        
        # Save files
        X_train_mrmr_save.to_csv(os.path.join(save_path, f"mrmr_X_train{dataset_suffix}.csv"), index=False)
        X_test_mrmr_save.to_csv(os.path.join(save_path, f"mrmr_X_test{dataset_suffix}.csv"), index=False)
        
        print(f"Saved selected features datasets with {len(selected_features)} features to {save_path}")
    
    return X_train_mrmr, X_test_mrmr, selected_features

    # Apply mRMR feature selection
    selected_data = mrmr_features(data_encoded, target='default payment next month', max_features=20)

if __name__ == "__main__":
    X_train = pd.read_csv("./data/processed/original_X_train.csv")
    X_test = pd.read_csv("./data/processed/original_X_test.csv")
    y_train = pd.read_csv("./data/processed/original_y_train.csv").values.ravel()

    # X_train_pca, X_test_pca = pca_features(X_train, X_test, scree_plot=True, save=True)
    X_train_mrmr, X_test_mrmr, selected_features = mrmr_features(X_train, X_test, y_train, plotQ=True, max_features=20, save_path="./data/processed/mRMR")