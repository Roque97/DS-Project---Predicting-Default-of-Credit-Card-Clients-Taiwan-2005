import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

numerical_features = [
        'LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
        'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2',
        'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
    ]

categorical_cols = [
        'SEX', 'EDUCATION', 'MARRIAGE',
        'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
        'default payment next month'
    ]

target = 'default payment next month'


def load_data(path='data/raw/default_of_credit_card_clients.xls'):
    return pd.read_excel(path, header=1)

def preprocess_data(
    data, 
    test_size=0.2, 
    random_state=12345, 
    split_data=True, 
    save=False, 
    save_prefix="original"
):
    """ Preprocess the credit card default dataset. This includes:
    - Converting data types for numerical and categorical features.
    - Mapping categorical values to more meaningful labels.
    - Handling missing values by dropping rows with any missing data.
    - Creating new features based on existing ones, particularly for payment history.
    - Optionally splitting data into train/test sets.
    
    Args:
        data (pd.DataFrame): Raw data loaded from the source.
        test_size (float): Proportion of data to use for testing (default: 0.2).
        random_state (int): Random state for reproducibility (default: 12345).
        split_data (bool): Whether to split data into train/test sets (default: True).
        save (bool): Whether to save the processed data to CSV files (default: False).
        save_prefix (str): Prefix for saved file names (default: "processed").
        
    Returns:
        dict: Dictionary containing processed data and splits if requested.
              Keys: 'original', 'transformed', 'encoded', and optionally 
              'X_train', 'X_test', 'y_train', 'y_test'
    """

    #Changing data types and handling missing values
    data[numerical_features] = data[numerical_features].apply(pd.to_numeric, errors='coerce')
    for col in categorical_cols:
        data[col] = data[col].astype('category')
    data['ID'] = data['ID'].astype(str)
    data['SEX'] = data['SEX'].map({1: 'male', 2: 'female'}).astype('category')
    data['EDUCATION'] = data['EDUCATION'].map({
        1: 'graduate school', 2: 'university', 3: 'high school', 4: 'others',
        5: np.nan, 6: np.nan, 0: np.nan
    }).astype('category')
    data['MARRIAGE'] = data['MARRIAGE'].map({
        1: 'married', 2: 'single', 3: 'divorce',
        0: np.nan
    }).astype('category')
    education_order = ['graduate school', 'high school', 'university', 'others']

    # Create ordered categories
    data['EDUCATION'] = data['EDUCATION'].cat.reorder_categories(education_order, ordered=True)
    marriage_order = ['single', 'married', 'divorce']
    data['MARRIAGE'] = data['MARRIAGE'].cat.reorder_categories(marriage_order, ordered=True)

    # Drop rows with any missing values
    data = data.dropna()
    print(f"Data shape after dropping missing values: {data.shape}")

    # List of PAY_X columns to transform and remove
    pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

    # Copy data for transformation
    new_data = data.copy()

    # Create new features
    for col in pay_cols:
        new_data[f'{col}_no_consumption'] = (new_data[col] == -2).astype(int)
        new_data[f'{col}_paid_duly'] = (new_data[col] == -1).astype(int)
        new_data[f'{col}_delay'] = new_data[col].apply(lambda x: 0 if x < 0 else x)
    new_data[target] = new_data[target].astype(int)

    # Remove original PAY_X columns after feature engineering
    new_data = new_data.drop(columns=pay_cols)

    # Exclude the target variable from dummyfication
    categorical_to_dummy = [col for col in new_data.select_dtypes(include='category').columns if col != target]

    # Perform one-hot encoding, keeping the target as category
    data_encoded = pd.get_dummies(new_data, columns=categorical_to_dummy, drop_first=False)

    # Prepare return dictionary
    result = {
        'original': data,
        'transformed': new_data,
        'encoded': data_encoded
    }

    # Split data if requested
    if split_data:
        # Prepare features and target
        feature_cols = [col for col in data_encoded.columns if col not in [target, 'ID']]
        X = data_encoded[feature_cols]
        y = data_encoded[target]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Add splits to result
        result.update({
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        })
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        print(f"Class distribution in training set:\n{y_train.value_counts(normalize=True)}")

    # Save results if requested
    if save:
        import os
        os.makedirs("data/processed", exist_ok=True)
        data.to_csv(f"data/processed/{save_prefix}_original.csv", index=False)
        new_data.to_csv(f"data/processed/{save_prefix}_transformed.csv", index=False)
        data_encoded.to_csv(f"data/processed/{save_prefix}_encoded.csv", index=False)
        if split_data:
            X_train.to_csv(f"data/processed/{save_prefix}_X_train.csv", index=False)
            X_test.to_csv(f"data/processed/{save_prefix}_X_test.csv", index=False)
            y_train.to_csv(f"data/processed/{save_prefix}_y_train.csv", index=False)
            y_test.to_csv(f"data/processed/{save_prefix}_y_test.csv", index=False)
        print("Processed files saved to data/processed/")

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
    # Engineered binary features
    pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    binary_features = []
    delay_features = []
    for col in pay_cols:
        binary_features += [f'{col}_no_consumption', f'{col}_paid_duly']
        delay_features += [f'{col}_delay']

    # Target and ID
    target = 'default payment next month'
    id_col = 'ID'

    # Set types
    for col in numerical_features + delay_features:
        if col in data_encoded.columns:
            data_encoded[col] = data_encoded[col].astype(float)
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

if __name__ == "__main__":
    data = load_data()
    
    # Example usage with splitting
    processed_with_split = preprocess_data(data, test_size=0.2, random_state=12345, split_data=True, save=True)