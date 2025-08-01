import pandas as pd

# Load the dataset
dataset_path = 'data/raw/default_of_credit_card_clients.xls'
data = pd.read_excel(dataset_path, header=1)

# Convert specific columns to categorical types
categorical_cols = [
    'SEX', 'EDUCATION', 'MARRIAGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'default payment next month'
]
for col in categorical_cols:
    data[col] = data[col].astype('category')

data['ID'] = data['ID'].astype(str)


# Map categorical codes to labels
data['SEX'] = data['SEX'].map({1: 'male', 2: 'female'}).astype('category')
data['EDUCATION'] = data['EDUCATION'].map({
    1: 'graduate school', 2: 'university', 3: 'high school', 4: 'others',
    5: 'unknown', 6: 'unknown', 0: 'unknown'
}).astype('category')
data['MARRIAGE'] = data['MARRIAGE'].map({1: 'married', 2: 'single', 3: 'others', 0: 'unknown'}).astype('category')