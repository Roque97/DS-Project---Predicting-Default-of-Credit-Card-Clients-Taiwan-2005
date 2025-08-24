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
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

from .feature_engineering import set_types_encoded, DataSplitDict

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
    X_test = data.X_test
    y_test = data.y_test

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

    

    # Standardize features (call method on data)
    nb = GaussianNB()
    nb.fit(data.X_train, data.y_train)

    results = evaluate_model(nb, data, plotsQ=plotsQ, save_path=save_path)
    return nb, results

def KNNClassifier(
    data: DataSplitDict,
    save_path: str = None,
    plotsQ: bool = True,
    param_grid: Optional[dict] = None
) -> Tuple[KNeighborsClassifier, dict]:
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

    X_train = data.X_train
    X_test = data.X_test
    y_train = data.y_train
    y_test = data.y_test

    if param_grid is None:
        param_grid = {'n_neighbors': test_cases_n if test_cases_n is not None else list(range(1, 40, 2))}
        
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv=10, scoring='f1')
    grid.fit(X_train, y_train)

    print("Best n_neighbors:", grid.best_params_['n_neighbors'])
    print("Best cross-validated F1 score:", grid.best_score_)

    if plotsQ:
        mean_scores = grid.cv_results_['mean_test_score']
        n_neighbors = param_grid['n_neighbors']
        plt.figure(figsize=(8, 5))
        plt.plot(n_neighbors, mean_scores, marker='o')
        plt.xlabel('Number of Neighbors (k)')
        plt.ylabel('Cross-Validated F1 Score')
        plt.title('KNN F1 Score vs Number of Neighbors')
        plt.grid(True)
        plt.show()

    best_knn = grid.best_estimator_
    results = evaluate_model(best_knn, data, plotsQ=plotsQ, save_path=save_path)
    return best_knn, results

def DecisionTreeClassifierModel(
    data: DataSplitDict,
    param_grid: Optional[dict] = None,
    save_path: str = None,
    plotsQ: bool = False
) -> Tuple[DecisionTreeClassifier, dict]:
    """
    Train and evaluate a Decision Tree classifier for credit card default prediction.
    Includes hyperparameter tuning using GridSearchCV.

    Parameters:
    - data: Dict with keys 'X_train', 'y_train', 'X_test', 'y_test'
    - param_grid: Dictionary of hyperparameters to search (default provides common DT parameters)
    - save_path: Directory to save plots
    - plotsQ: If True, generate and save plots

    Returns:
    - dt: Best trained Decision Tree model from grid search
    - results: Evaluation results dictionary
    """
    # Default parameter grid if none provided

    X_train = data.X_train
    X_test = data.X_test
    y_train = data.y_train
    y_test = data.y_test

    if param_grid is None:
        param_grid = {
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 5, 10],
            'criterion': ['gini', 'entropy']
        }
    
    # Create base model
    dt = DecisionTreeClassifier(random_state=12345)

    X_train = data.X_train.reset_index(drop=True)
    X_test = data.X_test.reset_index(drop=True)
    y_train = pd.Series(data.y_train).reset_index(drop=True)
    y_test = pd.Series(data.y_test).reset_index(drop=True)
    
    # Perform grid search
    grid = GridSearchCV(dt, param_grid, cv=5, scoring='recall')
    grid.fit(X_train, y_train)

    # Print best parameters and score
    print("Best parameters:", grid.best_params_)
    print("Best cross-validated F1 score:", grid.best_score_)
    
    if plotsQ:
        # Plot feature importance of best model
        best_dt = grid.best_estimator_
        importances = best_dt.feature_importances_
        features = X_train.columns
        indices = np.argsort(importances)[::-1]
        
        # Plot top 15 important features
        plt.figure(figsize=(10, 6))
        plt.bar([features[i] for i in indices[:15]], importances[indices[:15]])
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.title('Top Feature Importances (Decision Tree)')
        if save_path is not None:
            plt.savefig(os.path.join(save_path, 'feature_importance.png'))
        plt.show()
        
        # Plot the decision tree visualization
        plt.figure(figsize=(20, 10))
        # Limit max_depth for visualization if tree is too large
        # max_depth=3 shows the first few levels which is usually most informative
        visualization_depth = 3 if best_dt.get_depth() > 3 else None
        plot_tree(best_dt, 
                  feature_names=X_train.columns, 
                  class_names=['Not Default', 'Default'],
                  filled=True,
                  rounded=True,
                  fontsize=8,
                  max_depth=visualization_depth)
        plt.title('Decision Tree Visualization')
        if save_path is not None:
            plt.savefig(os.path.join(save_path, 'decision_tree_plot.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot max_depth vs accuracy if max_depth was in param_grid
        if 'max_depth' in param_grid and len(param_grid['max_depth']) > 1:
            results = grid.cv_results_
            depths = param_grid['max_depth']
            
            # Extract scores for different max_depths (averaging over other parameters)
            depth_scores = {}
            for depth in depths:
                if depth is None:
                    mask = [p['max_depth'] is None for p in results['params']]
                else:
                    mask = [p.get('max_depth') == depth for p in results['params']]
                depth_scores[str(depth)] = np.mean(results['mean_test_score'][mask])
            
            plt.figure(figsize=(8, 5))
            plt.plot(list(depth_scores.keys()), list(depth_scores.values()), marker='o')
            plt.xlabel('Max Depth')
            plt.ylabel('Cross-Validated Accuracy')
            plt.title('Decision Tree: Accuracy vs Max Depth')
            plt.grid(True)
            if save_path is not None:
                plt.savefig(os.path.join(save_path, 'max_depth_vs_accuracy.png'))
            plt.show()
    
    # Get best model and evaluate
    best_dt = grid.best_estimator_
    results = evaluate_model(best_dt, data, plotsQ=plotsQ, save_path=save_path)
    
    return best_dt, results

def RandomForestClassifierModel(
    data: DataSplitDict,
    param_grid: Optional[dict] = None,
    save_path: str = None,
    plotsQ: bool = False
) -> Tuple[DecisionTreeClassifier, dict]:
    """
    Train and evaluate a Random Forest classifier for credit card default prediction.
    Includes hyperparameter tuning using GridSearchCV.

    Parameters:
    - data: Dict with keys 'X_train', 'y_train', 'X_test', 'y_test'
    - param_grid: Dictionary of hyperparameters to search (default provides common RF parameters)
    - save_path: Directory to save plots
    - plotsQ: If True, generate and save plots

    Returns:
    - rf: Best trained Random Forest model from grid search
    - results: Evaluation results dictionary
    """

    X_train = data.X_train
    X_test = data.X_test
    y_train = data.y_train
    y_test = data.y_test
    
    # Default parameter grid if none provided
    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }
    
    # Create base model
    rf = RandomForestClassifier(random_state=12345)
    
    # Perform grid search
    grid = GridSearchCV(rf, param_grid, cv=5, scoring='recall')
    grid.fit(X_train, y_train)
    
    # Print best parameters and score
    print("Best parameters:", grid.best_params_)
    print("Best cross-validated recall:", grid.best_score_)
    
    if plotsQ:
        # Plot feature importance of best model
        best_rf = grid.best_estimator_
        importances = best_rf.feature_importances_
        features = X_train.columns
        indices = np.argsort(importances)[::-1]
        
        # Plot top 15 important features
        plt.figure(figsize=(10, 6))
        plt.title("Random Forest: Feature Importance")
        plt.barh(range(15), importances[indices][:15], align="center")
        plt.yticks(range(15), features[indices][:15])
        plt.xlabel("Feature Importance")
        if save_path is not None:
            plt.savefig(os.path.join(save_path, 'feature_importance.png'))
        plt.show()
        # Plot n_estimators vs accuracy if n_estimators was in param_grid
        if 'n_estimators' in param_grid and len(param_grid['n_estimators']) > 1:
            results = grid.cv_results_
            n_estimators = param_grid['n_estimators']
            
            # Extract scores for different n_estimators (averaging over other parameters)
            estimator_scores = {}
            for n_est in n_estimators:
                mask = [p.get('n_estimators') == n_est for p in results['params']]
                estimator_scores[n_est] = np.mean(results['mean_test_score'][mask])
            
            plt.figure(figsize=(8, 5))
            plt.plot(list(estimator_scores.keys()), list(estimator_scores.values()), marker='o')
            plt.xlabel('Number of Estimators')
            plt.ylabel('Cross-Validated Accuracy')
            plt.title('Random Forest: Accuracy vs Number of Estimators')
            plt.grid(True)
            if save_path is not None:
                plt.savefig(os.path.join(save_path, 'n_estimators_vs_accuracy.png'))
            plt.show()
    # Get best model and evaluate
    best_rf = grid.best_estimator_
    results = evaluate_model(best_rf, data, plotsQ=plotsQ, save_path=save_path)
    
    return best_rf, results


if __name__ == "__main__":
   
    print('Current working directory:', os.getcwd())
    original_data = DataSplitDict(
        X_train=pd.read_csv('data/processed/original_X_train.csv'),
        y_train=pd.read_csv('data/processed/original_y_train.csv').values.ravel(),
        X_test=pd.read_csv('data/processed/original_X_test.csv'),
        y_test=pd.read_csv('data/processed/original_y_test.csv').values.ravel()
    )

    original_data.X_train = set_types_encoded(original_data.X_train)
    original_data.X_test = set_types_encoded(original_data.X_test)
    print(original_data.X_train.head(20))

    # Standardize features
    original_data_standardized = original_data.standardize_features()

    print(original_data_standardized.X_train['AGE'].describe())

    print(original_data.X_train.head())

    # Naive Bayes Classifier
    
    #results_nb = NaiveBayesClassifier(original_data, save_path='plots/models/NaiveBayes/original/', plotsQ=True)
    
    # KNN Classifier
    #test_cases_n = list(range(1, 50, 2))  # Default test cases for KNN
    #KNNClassifier(original_data, test_cases_n, save_path='plots/models/KNN/original/')


