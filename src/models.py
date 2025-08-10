import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import os


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, plotsQ : bool = False, save_path: str = None):
    """
    Evaluate the model using various classification metrics on credit card default prediction.
    
    Parameters:
    - model: The trained model to evaluate
    - X_test: Test features DataFrame
    - y_test: Test labels Series (ground truth)
    - metrics: List of metric functions to evaluate the model (optional)
    
    Returns:
    - results: Dictionary with evaluation metrics and their values, which includes:
        - confusion_matrix: Confusion matrix of predictions
        - classification_report: Detailed classification report (precision, recall, f1-score)
        - actual_default_rate: Actual default rate in the test set
        - predicted_default_rate: Predicted default rate from the model
        - roc_curve: ROC curve data (fpr, tpr, thresholds)
        - roc_auc: Area Under the ROC Curve (AUC)
        - pr_curve: Precision-Recall curve data (precision, recall, thresholds)
        - avg_precision: Average precision score
    - If plotsQ is True, it will generate and save plots for confusion matrix, ROC curve, PR curve, and metrics table.
    - If save_path is provided, plots will be saved to that directory.
    """
    
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
            os.makedirs(save_path, exist_ok=True)

        # Plot confusion matrix with numbers
        plt.figure(figsize=(8, 6))
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
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