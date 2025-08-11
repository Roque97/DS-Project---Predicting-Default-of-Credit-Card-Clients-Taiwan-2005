# DS-Project---Predicting-Default-of-Credit-Card-Clients-Taiwan-2005

## Credit Card Default Prediction Project
This project aims to predict whether credit card clients will default on their payments using machine learning techniques. The dataset contains information on credit card clients in Taiwan from 2005, including demographic data, payment history, bill statements, and payment amounts.

"This research employed a binary variable, default payment (Yes = 1, No = 0), as the response variable. This study reviewed the literature and used the following 23 variables as explanatory variables:

X1: Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit.

X2: Gender (1 = male; 2 = female).

X3: Education (1 = graduate school; 2 = university; 3 = high school; 0, 4, 5, 6 = others).

X4: Marital status (1 = married; 2 = single; 3 = divorce; 0=others).

X5: Age (year).

X6 - X11: History of past payment. We tracked the past monthly payment records (from April to September, 2005) as follows: X6 = the repayment status in September, 2005; X7 = the repayment status in August, 2005; . . .;X11 = the repayment status in April, 2005. The measurement scale for the repayment status is:

-2: No consumption; -1: Paid in full; 0: The use of revolving credit; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.

X12-X17: Amount of bill statement (NT dollar). X12 = amount of bill statement in September, 2005; X13 = amount of bill statement in August, 2005; . . .; X17 = amount of bill statement in April, 2005.

X18-X23: Amount of previous payment (NT dollar). X18 = amount paid in September, 2005; X19 = amount paid in August, 2005; . . .;X23 = amount paid in April, 2005.

Y: client's behavior; Y=0 then not default, Y=1 then default"

Project Structure
.
├── data/
│   ├── processed/           # Preprocessed datasets
│   │   ├── MRMR/            # Features selected with mRMR
│   │   └── PCA/             # PCA transformed features
│   └── raw/                 # Original dataset
├── notebooks/
│   ├── eda.ipynb            # Exploratory Data Analysis
│   └── modeling.ipynb       # Model training and evaluation
├── plots/
│   ├── eda/                 # EDA visualizations
│   └── models/              # Model performance visualizations
├── reports/
│   └── tables/              # Performance summary tables
├── saved_models/            # Serialized trained models
├── src/
│   ├── __init__.py
│   ├── download_dataset.py  # Script to download the dataset
│   ├── feature_engineering.py # Feature engineering techniques
│   ├── models.py            # Model implementations
│   └── preprocessing.py     # Data preprocessing functions
├── .env                     # Environment variables
├── .gitignore               # Git ignore file
├── LICENSE                  # Apache 2.0 license
├── notes.txt                # Project notes
├── README.md                # Project documentation
└── requirements.txt         # Python dependencies

git clone https://github.com/yourusername/DS-Project---Predicting-Default-of-Credit-Card-Clients-Taiwan-2005.git
cd DS-Project---Predicting-Default-of-Credit-Card-Clients-Taiwan-2005
## Installation and Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/DS-Project---Predicting-Default-of-Credit-Card-Clients-Taiwan-2005.git
   cd DS-Project---Predicting-Default-of-Credit-Card-Clients-Taiwan-2005
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Download the dataset:
   ```
   python src/download_dataset.py
   ```

## Dataset Description

The dataset contains information on credit card clients with the following features:

| Feature | Description |
|---------|-------------|
| X1 | Amount of credit given (NT dollar) |
| X2 | Gender (1 = male; 2 = female) |
| X3 | Education (1 = graduate school; 2 = university; 3 = high school; 0, 4, 5, 6 = others) |
| X4 | Marital status (1 = married; 2 = single; 3 = divorce; 0 = others) |
| X5 | Age (years) |
| X6-X11 | History of past payment (April to September 2005) |
| X12-X17 | Amount of bill statement (April to September 2005, NT dollar) |
| X18-X23 | Amount of previous payment (April to September 2005, NT dollar) |
| Y | Default payment (1 = yes, 0 = no) |

### Payment Status Scale
- `-2`: No consumption
- `-1`: Paid in full
- `0`: Use of revolving credit
- `1-9`: Payment delay for n months (1 = one month, 9 = nine months and above)

## Methodology

### Preprocessing
- Data type conversion for categorical and numerical features
- Mapping categorical values to meaningful labels
- Handling missing values
- Feature engineering from payment history variables

### Feature Engineering
- **PCA**: Dimensionality reduction while preserving variance
- **mRMR**: Feature selection by maximizing relevance and minimizing redundancy

### Modeling
- **Naive Bayes**: Probabilistic classifier based on Bayes' theorem
- **K-Nearest Neighbors**: Classification based on k closest training examples
- **Decision Trees**: Tree-based model with interpretable decision rules

## Results

Our models achieve varying performance in predicting credit card defaults:
- The best performing model achieves around 78% accuracy on the test set
- Feature importance analysis shows payment history features are most predictive
- The imbalanced nature of the dataset (22% defaults) requires careful evaluation metrics

## Usage

### Exploratory Data Analysis
```
jupyter notebook notebooks/eda.ipynb
```

### Model Training and Evaluation
```
jupyter notebook notebooks/modeling.ipynb
```

## License

This project is licensed under the Apache License 2.0 - see the [`LICENSE`](LICENSE) file for details.