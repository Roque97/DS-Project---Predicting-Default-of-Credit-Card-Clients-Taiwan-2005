import pandas as pd
import numpy as np

numerical_features = [
        'LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
        'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2',
        'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
    ]

categorical_features = [
        'SEX', 'EDUCATION', 'MARRIAGE',
        'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
        'default payment next month'
    ]

target = 'default payment next month'


def load_data(path='data/raw/default_of_credit_card_clients.xls'):
    return pd.read_excel(path, header=1)

def preprocess_data(
    data : pd.DataFrame, 
    test_size: float = 0.2, 
    random_state: int = 12345, 
    split_data: bool = True, 
    save_prefix: str = None
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
        save_prefix (str): Prefix for saved file names (default: "None").
        
    Returns:
        dict: Dictionary containing processed data and splits if requested.
              Keys: 'original', 'transformed', 'encoded', and optionally 
              'X_train', 'X_test', 'y_train', 'y_test'
    """
    new_data = data.copy()

    #Changing data types and handling missing values
    new_data[numerical_features] = new_data[numerical_features].apply(pd.to_numeric, errors='coerce')
    for col in categorical_features:
        new_data[col] = new_data[col].astype('category')
    new_data['ID'] = new_data['ID'].astype(str)
    new_data['SEX'] = new_data['SEX'].map({1: 'male', 2: 'female'}).astype('category')
    new_data['EDUCATION'] = new_data['EDUCATION'].map({
        1: 'graduate school', 2: 'university', 3: 'high school', 4: 'others',
        5: np.nan, 6: np.nan, 0: np.nan
    }).astype('category')
    new_data['MARRIAGE'] = new_data['MARRIAGE'].map({
        1: 'married', 2: 'single', 3: 'divorce',
        0: np.nan
    }).astype('category')
    education_order = ['high school', 'university', 'graduate school', 'others']

    # Create ordered categories
    new_data['EDUCATION'] = new_data['EDUCATION'].cat.reorder_categories(education_order, ordered=True)
    marriage_order = ['single', 'married', 'divorce']
    new_data['MARRIAGE'] = new_data['MARRIAGE'].cat.reorder_categories(marriage_order, ordered=True)

    # Drop rows with any missing values
    new_data = new_data.dropna()
    print(f"Data shape after dropping missing values: {new_data.shape}")

    # List of PAY_X columns to transform and remove
    pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

    return new_data

    ############################

    
if __name__ == "__main__":
    data = load_data()
    preprocessed_data = preprocess_data(data)
    print(preprocessed_data.head())