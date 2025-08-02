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



    return data

if __name__ == "__main__":
    data = load_data()
    data = preprocess_data(data)