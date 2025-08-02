import pandas as pd
import numpy as np

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

def preprocess_data(data):
    """ Preprocess the credit card default dataset. This includes:
    - Converting data types for numerical and categorical features.
    - Mapping categorical values to more meaningful labels.
    - Handling missing values by dropping rows with any missing data.
    - Creating new features based on existing ones, particularly for payment history.
    
    Args:
        data (pd.DataFrame): Raw data loaded from the source.
    Returns:
        list: A list containing the original data, transformed data, and one-hot encoded data."""

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

    # Exclude the target variable from dummyfication
    categorical_to_dummy = [col for col in new_data.select_dtypes(include='category').columns if col != target]

    # Perform one-hot encoding, keeping the target as category
    data_encoded = pd.get_dummies(new_data, columns=categorical_to_dummy, drop_first=False)

    return [data, new_data, data_encoded]

if __name__ == "__main__":
    data = load_data()
    processed_data = preprocess_data(data)[2]