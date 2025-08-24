import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from feature_engine.selection import MRMR
import os

from .preprocessing import load_data, preprocess_data
from imblearn.over_sampling import SMOTE

target = 'default payment next month'


def create_new_features(data: pd.DataFrame) -> dict:
    """Create new features based on payment history in the credit card default dataset.
    This includes:
    - Creating binary features for payment status (no consumption, paid duly, delay).
    - Creating delay features for each payment month.
    - Removing original PAY_X columns after feature engineering.
    
    Args:
        data (pd.DataFrame): Raw or preprocessed data.
        
    Returns:
        dict: Dictionary with 'transformed' and 'encoded' DataFrames.
    """
    # Handle the PAY_0/PAY_1 naming issue
    new_data = data.copy()
    
    # Check if PAY_0 exists but PAY_1 doesn't - rename if needed
    if 'PAY_0' in new_data.columns and 'PAY_1' not in new_data.columns:
        print("Renaming PAY_0 to PAY_1 for consistency")
        new_data = new_data.rename(columns={'PAY_0': 'PAY_1'})
    
    # Define payment columns (now we're sure PAY_1 exists if either was present)
    pay_cols = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    
    # Verify that all required columns exist
    missing_cols = [col for col in pay_cols if col not in new_data.columns]
    if missing_cols:
        raise ValueError(f"Missing required payment columns: {missing_cols}. Please check your data.")

    # Create new features
    for col in pay_cols:
        new_data[f'{col}_no_consumption'] = (new_data[col] == -2).astype(bool)
        new_data[f'{col}_paid_duly'] = (new_data[col] == -1).astype(bool)
        new_data[f'{col}_delay'] = new_data[col].apply(lambda x: 0 if x < 0 else x).astype(int)
    
    # Remove original PAY_X columns after feature engineering
    new_data = new_data.drop(columns=pay_cols)

    # Create avg_bill feature
    bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    if all(col in new_data.columns for col in bill_cols):
        new_data['avg_bill'] = new_data[bill_cols].mean(axis=1).astype(float)
    
    # Convert target to boolean instead of category to avoid category ordering issues
    if target in new_data.columns:
        new_data[target] = new_data[target].astype(int)

    # Get categorical columns but exclude target which we'll handle separately
    categorical_to_dummy = []
    for col in new_data.select_dtypes(include=['category', 'object']).columns:
        if col != target and col != 'ID':
            categorical_to_dummy.append(col)

    # Handle categorical columns safely:
    # 1. Convert to string first (removes category ordering that might cause issues)
    # 2. Then perform one-hot encoding
    for col in categorical_to_dummy:
        new_data[col] = new_data[col].astype(str)
    
    # Perform one-hot encoding
    data_encoded = pd.get_dummies(new_data, columns=categorical_to_dummy, drop_first=False)

    # Ensure target has correct type in encoded data
    if target in data_encoded.columns:
        data_encoded[target] = data_encoded[target].astype(int)

    # Prepare return dictionary
    result = {
        'transformed': new_data,
        'encoded': data_encoded
    }

    return result

def split_data(data_encoded: pd.DataFrame, target: str = None, test_size: float = 0.2, random_state: int = 12345, save_prefix: str = None) -> dict:
    """
    Split the encoded data into training and test sets.
    
    Args:
        data_encoded (pd.DataFrame): The encoded DataFrame to split
        target (str, optional): Target column name. Defaults to 'default payment next month'
        test_size (float): Proportion of test set. Defaults to 0.2
        random_state (int): Random state for reproducibility. Defaults to 12345
        save_prefix (str, optional): Prefix for saving files. If None, files won't be saved
        
    Returns:
        dict: Dictionary containing X_train, X_test, y_train, y_test
    """
    # Use default target if not provided
    if target is None:
        target = 'default payment next month'
    
    # Prepare features and target
    feature_cols = [col for col in data_encoded.columns if col not in [target, 'ID']]
    X = data_encoded[feature_cols]
    y = data_encoded[target]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Create result dictionary
    result = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Class distribution in training set:\n{y_train.value_counts(normalize=True)}")

    # Save results if requested
    if save_prefix:
        os.makedirs("data/processed", exist_ok=True)
        X_train.to_csv(f"data/processed/{save_prefix}_X_train.csv", index=False)
        X_test.to_csv(f"data/processed/{save_prefix}_X_test.csv", index=False)
        y_train.to_csv(f"data/processed/{save_prefix}_y_train.csv", index=False)
        y_test.to_csv(f"data/processed/{save_prefix}_y_test.csv", index=False)
        print(f"Processed files saved to data/processed/ with prefix '{save_prefix}'")

    return result

def set_types_encoded(data_encoded: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure that after loading the encoded dataset from CSV, the column types are set correctly.
    - Numerical columns are set to float.
    - Binary/engineered columns are set to int.
    - Target column is set to int.
    - ID column is set to string.
    - One-hot encoded categorical columns are set to int.

    Args:
        data_encoded (pd.DataFrame): The DataFrame loaded from CSV.

    Returns:
        pd.DataFrame: DataFrame with correct types.
    """
    # Numerical features
    numerical_features = [
        'LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
        'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2',
        'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
    ]

    original_categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']


    # Engineered binary features
    pay_cols = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

    binary_features = []
    delay_features = []
    for col in pay_cols:
        binary_features += [f'{col}_no_consumption', f'{col}_paid_duly']
        delay_features += [f'{col}_delay']


    #Add binary features for original categorical features by searching columns that start with the original feature name
    for col in data_encoded.columns:
        for orig_col in original_categorical_features:
            if col.startswith(f"{orig_col}_"):
                binary_features.append(col)

    # Target and ID
    target = 'default payment next month'
    id_col = 'ID'

    # Set types
    for col in numerical_features:
        if col in data_encoded.columns:
            data_encoded[col] = data_encoded[col].astype(float)

    for col in delay_features:
        if col in data_encoded.columns:
            data_encoded[col] = data_encoded[col].astype(int)
    for col in binary_features:
        if col in data_encoded.columns:
            data_encoded[col] = data_encoded[col].astype(bool)
    if target in data_encoded.columns:
        data_encoded[target] = data_encoded[target].astype(int)
    if id_col in data_encoded.columns:
        data_encoded[id_col] = data_encoded[id_col].astype(str)

    # One-hot encoded categorical columns (all remaining object columns except ID)
    for col in data_encoded.select_dtypes(include='object').columns:
        if col != id_col:
            data_encoded[col] = data_encoded[col].astype(int)
    return data_encoded

class DataSplitDict:
    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def standardize_features(self) -> "DataSplitDict":
        """
        Return a new DataSplitDict with standardized numerical features in X_train and X_test.

        Returns:
            DataSplitDict: A new DataSplitDict with standardized numerical features.
        """
        import copy
        X_train_std = self.X_train.copy()
        X_test_std = self.X_test.copy()
        y_train = self.y_train.copy() if hasattr(self.y_train, 'copy') else self.y_train
        y_test = self.y_test.copy() if hasattr(self.y_test, 'copy') else self.y_test

        numerical_features = X_train_std.select_dtypes(include=[np.float64, np.int64]).columns.tolist()
        scaler = StandardScaler()

        X_train_std[numerical_features] = scaler.fit_transform(X_train_std[numerical_features])
        X_test_std[numerical_features] = scaler.transform(X_test_std[numerical_features])

        return DataSplitDict(X_train_std, y_train, X_test_std, y_test)
    
    def oversample_smote(self) -> "DataSplitDict":
        """
        Return a new DataSplitDict with SMOTE oversampling applied to the training data.

        Returns:
            DataSplitDict: A new DataSplitDict with SMOTE applied to the training data.
        """

        X_train_os = self.X_train.copy()
        y_train_os = self.y_train.copy() if hasattr(self.y_train, 'copy') else self.y_train
        X_test_os = self.X_test.copy()
        y_test_os = self.y_test.copy() if hasattr(self.y_test, 'copy') else self.y_test

        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train_os, y_train_os)

        return DataSplitDict(X_resampled, y_resampled, X_test_os, y_test_os)

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

    return X_train_pca, X_test_pca, pca

def mrmr_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    max_features: int = 20,
    plotQ: bool = False,
    save_path: str = None,
    dataset_suffix: str = ""
) -> tuple:
    """
    Select features using mRMR (Minimum Redundancy Maximum Relevance).
    Always includes 'avg_bill' feature if available and returns exactly max_features features.
    
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
    
    # Check if avg_bill exists in the dataset
    has_avg_bill = 'avg_bill' in X_train_features.columns
    
    # If avg_bill exists, we need to ensure it's included in the final selection
    if has_avg_bill:
        # We'll select one less feature with mRMR to make room for avg_bill
        # if avg_bill doesn't get selected automatically
        mrmr_max_features = max_features - 1
        print(f"Reserving space for 'avg_bill', setting mRMR to select {mrmr_max_features} features")
    else:
        mrmr_max_features = max_features
    
    # Ensure mrmr_max_features isn't larger than the number of available features
    mrmr_max_features = min(mrmr_max_features, X_train_features.shape[1])
    print(f"Using max_features={mrmr_max_features} for mRMR selection, total features: {X_train_features.shape[1]}")
    
    selected_features = []
    try:
        # Try with MRMR
        mrmr_selector = MRMR(
            variables=None,
            max_features=mrmr_max_features,
            scoring='recall',
            random_state=12345
        )
        
        # Fit on training data
        print("Fitting MRMR selector...")
        mrmr_selector.fit(X_train_features, y_train)
        
        # Get selected features
        selected_features = X_train_features.columns[mrmr_selector.get_support()].tolist()
        
        # Check if avg_bill was selected
        if has_avg_bill:
            if 'avg_bill' not in selected_features:
                # avg_bill wasn't selected, so we add it
                selected_features.append('avg_bill')
                print("Adding 'avg_bill' to selected features")
            else:
                # avg_bill was already selected, so we need to select one more feature
                # to reach exactly max_features
                if len(selected_features) < max_features:
                    print("avg_bill was already selected, adding one more feature")
                    # Get non-selected features
                    remaining_features = [f for f in X_train_features.columns 
                                          if f not in selected_features]
                    if remaining_features:
                        # Calculate mutual information to find the best additional feature
                        from sklearn.feature_selection import mutual_info_classif
                        mi_scores = mutual_info_classif(
                            X_train_features[remaining_features], 
                            y_train, 
                            random_state=12345
                        )
                        # Get the feature with highest MI score
                        best_additional = remaining_features[mi_scores.argmax()]
                        selected_features.append(best_additional)
                        print(f"Added '{best_additional}' based on mutual information")
        
        # If we have too many features (shouldn't happen, but just in case)
        if len(selected_features) > max_features:
            # Keep avg_bill and remove the least important features
            if has_avg_bill and 'avg_bill' in selected_features:
                # Remove avg_bill temporarily
                selected_features.remove('avg_bill')
                # Calculate importance of remaining features
                from sklearn.feature_selection import mutual_info_classif
                mi_scores = mutual_info_classif(
                    X_train_features[selected_features], 
                    y_train, 
                    random_state=12345
                )
                # Sort features by importance
                feature_importance = sorted(zip(selected_features, mi_scores), 
                                            key=lambda x: x[1], reverse=True)
                # Keep only the most important ones up to max_features-1
                selected_features = [f[0] for f in feature_importance[:max_features-1]]
                # Add avg_bill back
                selected_features.append('avg_bill')
            else:
                # No avg_bill, just keep the most important features
                from sklearn.feature_selection import mutual_info_classif
                mi_scores = mutual_info_classif(
                    X_train_features[selected_features], 
                    y_train, 
                    random_state=12345
                )
                # Sort features by importance
                feature_importance = sorted(zip(selected_features, mi_scores), 
                                            key=lambda x: x[1], reverse=True)
                # Keep only the most important ones up to max_features
                selected_features = [f[0] for f in feature_importance[:max_features]]
        
        # If we still don't have enough features (e.g., if the dataset has fewer features than max_features)
        while len(selected_features) < max_features and len(selected_features) < len(X_train_features.columns):
            # Get non-selected features
            remaining_features = [f for f in X_train_features.columns 
                                  if f not in selected_features]
            if not remaining_features:
                break
                
            # Calculate mutual information to find the best additional feature
            from sklearn.feature_selection import mutual_info_classif
            mi_scores = mutual_info_classif(
                X_train_features[remaining_features], 
                y_train, 
                random_state=12345
            )
            # Get the feature with highest MI score
            best_additional = remaining_features[mi_scores.argmax()]
            selected_features.append(best_additional)
            print(f"Added '{best_additional}' to reach {max_features} features")
            
    except Exception as e:
        print(f"MRMR selection failed with error: {str(e)}")
        # Fallback to mutual information selection
        print("Falling back to mutual information selection")
        from sklearn.feature_selection import mutual_info_classif
        mi_scores = mutual_info_classif(X_train_features, y_train, random_state=12345)
        feature_importance = sorted(zip(X_train_features.columns, mi_scores), 
                                    key=lambda x: x[1], reverse=True)
        
        # If avg_bill exists, ensure it's included
        if has_avg_bill:
            # Get the top features except avg_bill
            top_features = [f[0] for f in feature_importance if f[0] != 'avg_bill']
            # Include at most max_features-1 features plus avg_bill
            selected_features = top_features[:max_features-1] + ['avg_bill']
        else:
            # Just get the top features
            selected_features = [f[0] for f in feature_importance[:max_features]]
    
    # Create the transformed datasets with only the selected features
    X_train_mrmr = X_train_features[selected_features].copy()
    X_test_mrmr = X_test_features[selected_features].copy()
    
    print(f"Final selected features count: {len(selected_features)}")
    print(f"avg_bill included: {'avg_bill' in selected_features}")
    print(f"Selected features: {selected_features}")
    
    # Plotting if requested
    if plotQ:
        try:
            # Calculate mutual information for visualization
            from sklearn.feature_selection import mutual_info_classif
            mi_scores = mutual_info_classif(
                X_train_features[selected_features], 
                y_train, 
                random_state=12345
            )
            features = selected_features
            
            plt.figure(figsize=(15, 4))
            plt.bar(features, mi_scores, color='skyblue')
            plt.xticks(rotation=90)
            plt.title("Feature Relevance Scores")
            plt.xlabel("Features")
            plt.ylabel("Mutual Information Score")
            plt.tight_layout()
            plt.show()
            
            # Plot top selected features' relevance
            selected_scores = dict(zip(features, mi_scores))
            pd.Series(selected_scores).sort_values(ascending=False).plot.bar(figsize=(12, 4))
            plt.title("Relevance of Selected Features")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Plotting error: {str(e)}")
            # Fallback plot
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


#run to create all the datasets
if __name__ == "__main__":
    # Load raw data
    data = load_data()
    
    # Preprocess the data first (this handles the PAY_0 -> PAY_1 rename and other preprocessing)
    preprocessed_data = preprocess_data(data, save_prefix="preprocessed")

    print(preprocessed_data.dtypes)
    
    # Now create new features using the preprocessed data
    df = create_new_features(preprocessed_data)['encoded']
    df = set_types_encoded(df)
    
    
    #split the data and save the splits
    splitted_data = split_data(df, save_prefix="original")
    data_dict = DataSplitDict(
        X_train=splitted_data['X_train'],
        y_train=splitted_data['y_train'],
        X_test=splitted_data['X_test'],
        y_test=splitted_data['y_test']
    )

    print(data_dict.X_train.dtypes)

    #Create PCA datasets
    # pca_X_train, pca_X_test = pca_features(
    #     X_train=data_dict.X_train,
    #     X_test=data_dict.X_test,
    #     save=True,
    #     scree_plot=True
    # )

    #Create mRMR datasets
    # mrmr_X_train, mrmr_X_test, selected_features = mrmr_features(
    #     X_train=data_dict.X_train,
    #     X_test=data_dict.X_test,
    #     y_train=data_dict.y_train,
    #     max_features=15,
    #     plotQ=True,
    #     save_path="./data/processed/mRMR",
    #     dataset_suffix=""
    # )   

    mrmr_X_train, mrmr_X_test, selected_features = mrmr_features(
        X_train=data_dict.X_train,
        X_test=data_dict.X_test,
        y_train=data_dict.y_train,
        max_features=15,
        plotQ=True
    )   






    print('Selected features:', selected_features)

    # If you want to create train/test splits, uncomment below:
    # split_result = split_data(df, save_prefix="engineered")

    # Example of running PCA and mRMR