import pandas as pd
import numpy as np
from typing import TypedDict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


class DataSplitDict(TypedDict):
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def standardize_features(data : pd.DataFrame) -> pd.DataFrame:
    """Standardize numerical features in the DataFrame.

    Args:
        data (pd.DataFrame): The DataFrame containing the dataset.

    Returns:
        pd.DataFrame: The DataFrame with standardized numerical features.
    """    
    # Identify numerical features for standardization
    numerical_features = data.select_dtypes(include=[np.float64, np.int64]).columns.tolist()

    for col in data.columns:
        if np.issubdtype(data[col].dtype, np.number) and col in numerical_features:
            scaler = StandardScaler()
            data[[col]] = scaler.fit_transform(data[[col]])
    return data


def evaluate_model(model, data: DataSplitDict, plotsQ : bool = False, save_path: str = None):
    """
    Evaluate the model using various classification metrics on credit card default prediction.

    Parameters:
    - model: The trained model to evaluate
    - data: DataSplitDict with keys 'X_test', 'y_test'
    - plotsQ: If True, generate and save plots
    - save_path: Directory to save plots

    Returns:
    - results: Dictionary with evaluation metrics and their values
    """
    X_test = data['X_test']
    y_test = data['y_test']

    results = {}

    # Get predictions
    y_pred = model.predict(X_test)

    # If model has predict_proba method, get probability predictions for ROC AUC, etc.
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]  # Probability of default (class 1)
    else:
        y_prob = None

    # Add confusion matrix
    results['confusion_matrix'] = confusion_matrix(y_test, y_pred)

    # Add classification report (precision, recall, f1 for each class)
    results['classification_report'] = classification_report(y_test, y_pred)

    # Calculate class distribution in predictions vs actual
    results['actual_default_rate'] = y_test.mean()
    results['predicted_default_rate'] = y_pred.mean()

    # Add ROC curve and AUC if probabilities are available
    if y_prob is not None:
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        results['roc_curve'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
        results['roc_auc'] = roc_auc_score(y_test, y_prob)

        # Add PR curve and average precision (useful for imbalanced data)
        precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
        results['pr_curve'] = {'precision': precision, 'recall': recall, 'thresholds': thresholds}
        results['avg_precision'] = average_precision_score(y_test, y_prob)

    if plotsQ:
        # If save_path is provided, create directory if needed
        if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)

        # Plot confusion matrix with numbers and percentages
        plt.figure(figsize=(8, 6))
        cm = results['confusion_matrix']
        cm_sum = np.sum(cm)
        annot = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                percent = cm[i, j] / cm_sum * 100
                annot[i, j] = f"{cm[i, j]}\n{percent:.1f}%"
        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                    xticklabels=['Not Default', 'Default'],
                    yticklabels=['Not Default', 'Default'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        if save_path is not None:
            plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
        plt.show()

        # Plot ROC curve
        if y_prob is not None:
            plt.figure(figsize=(8, 6))
            fpr = results['roc_curve']['fpr']
            tpr = results['roc_curve']['tpr']
            plt.plot(fpr, tpr, color='blue', label='ROC curve (area = {:.2f})'.format(results['roc_auc']))
            plt.plot([0, 1], [0, 1], color='red', linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC)')
            plt.legend(loc='lower right')
            if save_path is not None:
                plt.savefig(os.path.join(save_path, 'roc_curve.png'))
            plt.show()

        # Plot Precision-Recall curve
        if y_prob is not None:
            plt.figure(figsize=(8, 6))
            precision = results['pr_curve']['precision']
            recall = results['pr_curve']['recall']
            plt.plot(recall, precision, color='blue', label='Precision-Recall curve (AP = {:.2f})'.format(results['avg_precision']))
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc='lower left')
            if save_path is not None:
                plt.savefig(os.path.join(save_path, 'precision_recall_curve.png'))
            plt.show()

    # Plot metrics report as a table
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Value': [acc, prec, rec, f1]
    })
    plt.figure(figsize=(5, 2))
    plt.table(cellText=metrics_df.values, colLabels=metrics_df.columns, loc='center', cellLoc='center')
    plt.axis('off')
    plt.title('Classification Metrics')
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'metrics_table.png'))
    plt.show()
    return results

def NaiveBayesClassifier(
    data: DataSplitDict,
    save_path: str = None,
    plotsQ: bool = False
) -> Tuple[GaussianNB, dict]:
    """
    Train and evaluate a Naive Bayes classifier for credit card default prediction.

    Parameters:
    - data: Dict with keys 'X_train', 'y_train', 'X_test', 'y_test'
    - save_path (optional): Directory to save plots

    Returns:
    - nb: Trained Naive Bayes model
    - results: Evaluation results dictionary
    """

    

    # Standardize features
    X_train = standardize_features(data['X_train'])
    X_test = standardize_features(data['X_test'])

    nb = GaussianNB()
    nb.fit(X_train, data['y_train'])

    results = evaluate_model(nb, data, plotsQ=plotsQ, save_path=save_path)
    return nb, results

def KNNClassifier(
    data: DataSplitDict,
    test_cases_n: Optional[List[int]] = None,
    save_path: str = None,
    plotsQ: bool = True ) -> Tuple[KNeighborsClassifier, dict]:
    """
    Apply K-Nearest Neighbors (KNN) for credit card default prediction.

    Parameters:
    - data: Dict with keys 'X_train', 'y_train', 'X_test', 'y_test'
    - test_cases_n: List of n_neighbors values to try (default: 1-20)
    - save_path: Directory to save plots
    - plotsQ: If True, generate and save plots

    Returns:
    - best_knn: Best KNN model found
    - results: Evaluation results dictionary
    """



    param_grid = {'n_neighbors': test_cases_n if test_cases_n is not None else list(range(1, 21))}
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
    grid.fit(data['X_train'], data['y_train'])

    print("Best n_neighbors:", grid.best_params_['n_neighbors'])
    print("Best cross-validated accuracy:", grid.best_score_)

    if plotsQ:
        mean_scores = grid.cv_results_['mean_test_score']
        n_neighbors = param_grid['n_neighbors']
        plt.figure(figsize=(8, 5))
        plt.plot(n_neighbors, mean_scores, marker='o')
        plt.xlabel('Number of Neighbors (k)')
        plt.ylabel('Cross-Validated Accuracy')
        plt.title('KNN Accuracy vs Number of Neighbors')
        plt.grid(True)
        plt.show()

    best_knn = grid.best_estimator_
    results = evaluate_model(best_knn, data, plotsQ=plotsQ, save_path=save_path)
    return best_knn, results

def DecisionTreeClassifierModel(
    data: DataSplitDict,
    save_path: str = None,
    plotsQ: bool = False
) -> Tuple[DecisionTreeClassifier, dict]:
    """
    Train and evaluate a Decision Tree classifier for credit card default prediction.

    Parameters:
    - data: Dict with keys 'X_train', 'y_train', 'X_test', 'y_test'
    - save_path (optional): Directory to save plots

    Returns:
    - dt: Trained Decision Tree model
    - results: Evaluation results dictionary
    """
    dt = DecisionTreeClassifier(random_state=12345)
    dt.fit(data['X_train'], data['y_train'])
    results = evaluate_model(dt, data, plotsQ=plotsQ, save_path=save_path)
    return dt, results

if __name__ == "__main__":
   
    print('Current working directory:', os.getcwd())
    original_data = DataSplitDict(
        X_train=pd.read_csv('data/processed/original_X_train.csv'),
        y_train=pd.read_csv('data/processed/original_y_train.csv').values.ravel(),
        X_test=pd.read_csv('data/processed/original_X_test.csv'),
        y_test=pd.read_csv('data/processed/original_y_test.csv').values.ravel()
    )

    # Standardize features
    original_data['X_train'] = standardize_features(original_data['X_train'])
    original_data['X_test'] = standardize_features(original_data['X_test'])
    
    # Naive Bayes Classifier
    
    #results_nb = NaiveBayesClassifier(original_data, save_path='plots/models/NaiveBayes/original/', plotsQ=True)
    
    test_cases_n = list(range(1, 50, 2))  # Default test cases for KNN
    KNNClassifier(original_data, test_cases_n, save_path='plots/models/KNN/original/')


