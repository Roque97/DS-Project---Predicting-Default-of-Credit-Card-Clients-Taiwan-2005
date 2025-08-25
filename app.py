import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_curve, auc, precision_recall_curve
)

# Fix joblib import (sklearn.externals.joblib is deprecated)
try:
    import joblib
except ImportError:
    from sklearn.externals import joblib

# Import custom modules
sys.path.append('.')  # Add current directory to path
from src.preprocessing import preprocess_data
from src.feature_engineering import create_new_features, set_types_encoded

# Set page config
st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_model(model_path):
    """Load a pickled model from path"""
    try:
        with open(model_path, 'rb') as file:
            model_data = pickle.load(file)
            
        # Check if we loaded a dict with model and transformers or just a model
        if isinstance(model_data, dict) and 'model' in model_data:
            return model_data
        else:
            return {'model': model_data}
    except:
        try:
            return {'model': joblib.load(model_path)}
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

def apply_feature_engineering(data, model_data=None, pca_model_path=None, scaler_model_path=None):
    """Apply the same feature engineering used during training"""
    try:
        # Check if data is already preprocessed by looking for engineered features
        already_preprocessed = any(col.startswith('PAY_') and ('_delay' in col or '_no_consumption' in col or '_paid_duly' in col) 
                                 for col in data.columns)
        
        if already_preprocessed:
            st.info("📝 Input data appears to be already preprocessed. Skipping preprocessing steps.")
            # If already preprocessed, we just need to ensure types are correct
            data_encoded = set_types_encoded(data)
        else:
            # Apply the full preprocessing pipeline
            st.info("🔄 Applying preprocessing and feature engineering to raw data.")
            # Step 1: Preprocess the data
            preprocessed_data = preprocess_data(data, split_data=False)
            
            # Step 2: Create engineered features
            data_dict = create_new_features(preprocessed_data)
            data_encoded = data_dict['encoded']
            
            # Step 3: Ensure correct column types
            data_encoded = set_types_encoded(data_encoded)
        
        # Step 4: Check if we need to apply PCA
        if pca_model_path and scaler_model_path:
            # Load PCA and scaler models
            with open(pca_model_path, 'rb') as f:
                pca = pickle.load(f)
            with open(scaler_model_path, 'rb') as f:
                scaler = pickle.load(f)
                
            # Get expected feature names from scaler
            if hasattr(scaler, 'feature_names_in_'):
                expected_features = list(scaler.feature_names_in_)
                st.info(f"PCA model expects {len(expected_features)} features")
                
                # Create a DataFrame with zeros for all expected features
                X = pd.DataFrame(0, index=range(len(data_encoded)), columns=expected_features)
                
                # Copy available features from data_encoded to X
                common_features = [f for f in expected_features if f in data_encoded.columns]
                st.info(f"Found {len(common_features)} matching features out of {len(expected_features)} expected")
                
                # Fill in values for features that exist in our data
                for feature in common_features:
                    X[feature] = data_encoded[feature].astype(float)
                
                # IMPORTANT: Ensure columns are in the exact order the PCA model expects
                X = X[expected_features]
                
                # Ensure no NaNs
                X = X.fillna(0)
                
                # Scale features using the saved scaler
                X_scaled = scaler.transform(X)
                
                # Apply PCA transformation
                X_pca = pca.transform(X_scaled)
                
                # Return PCA transformed data
                pca_cols = [f'PC{i+1}' for i in range(X_pca.shape[1])]
                return pd.DataFrame(X_pca, columns=pca_cols)
            else:
                st.error("Scaler doesn't have feature_names_in_ attribute. Cannot align features for PCA.")
                return None
                
        # If no PCA is required, return the encoded data
        return data_encoded
    except Exception as e:
        st.error(f"Error in feature engineering: {e}")
        st.exception(e)
        return None

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix with percentages"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    
    # Create annotations with counts and percentages
    cm_sum = np.sum(cm)
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            percent = cm[i, j] / cm_sum * 100
            annot[i, j] = f"{cm[i, j]}\n{percent:.1f}%"
    
    ax = sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                xticklabels=['Not Default', 'Default'],
                yticklabels=['Not Default', 'Default'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    return plt

def plot_roc_curve(y_true, y_prob):
    """Plot ROC curve with AUC"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    
    return plt, roc_auc

def plot_precision_recall_curve(y_true, y_prob):
    """Plot Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = np.mean(precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='upper right')
    
    return plt

def calculate_metrics(y_true, y_pred, y_prob=None):
    """Calculate and return common classification metrics"""
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred)
    }
    
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        metrics["AUC"] = auc(fpr, tpr)
    
    return metrics

def display_feature_importance(model, feature_names):
    """Display feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Get top 15 features
        top_n = min(15, len(feature_names))
        top_indices = indices[:top_n]
        
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importance')
        plt.barh(range(top_n), importances[top_indices], align="center")
        plt.yticks(range(top_n), [feature_names[i] for i in top_indices])
        plt.xlabel('Feature Importance')
        plt.gca().invert_yaxis()  # Highest importance at the top
        
        return plt
    return None

def main():
    st.title("💳 Credit Card Default Prediction")
    st.write("""
    This app predicts the probability of credit card default based on customer information.
    Upload a CSV file with customer data to get predictions.
    """)
    
    # Sidebar for model selection
    st.sidebar.header("⚙️ Settings")
    
    # Get list of available models
    model_dir = './saved_models'
    if not os.path.exists(model_dir):
        st.sidebar.error(f"No models found in {model_dir}. Please ensure models are saved in this directory.")
        model_files = []
    else:
        model_files = [f for f in os.listdir(model_dir) if f.endswith(('.pkl', '.joblib')) 
                      and not (f.startswith('pca') or f.startswith('scaler'))]
    
    # PCA and scaler files
    pca_file = 'pca_model.pkl' if 'pca_model.pkl' in os.listdir(model_dir) else None
    scaler_file = 'scaler_model_pca.pkl' if 'scaler_model_pca.pkl' in os.listdir(model_dir) else None
    
    # Detect model types from filenames
    model_types = {}
    for file in model_files:
        if "NB" in file:
            model_types[file] = "Naive Bayes"
        elif "KNN" in file:
            model_types[file] = "K-Nearest Neighbors"
        elif "DT" in file or "decision_tree" in file.lower():
            model_types[file] = "Decision Tree"
        elif "RF" in file or "random_forest" in file.lower():
            model_types[file] = "Random Forest"
        elif "XGB" in file or "xgboost" in file.lower():
            model_types[file] = "XGBoost"
        else:
            model_types[file] = "Unknown Model"
    
    if not model_files:
        st.sidebar.warning("No models found. Please add models to the saved_models directory.")
        selected_model_path = None
    else:
        # Add model type to selection box
        model_options = [f"{model_types[file]} ({file})" for file in model_files]
        selected_model_option = st.sidebar.selectbox("Select a model", model_options)
        selected_model_file = selected_model_option.split("(")[1].split(")")[0]
        selected_model_path = os.path.join(model_dir, selected_model_file)
        
        st.sidebar.write(f"Selected model type: **{model_types[selected_model_file]}**")
        
        # Check if model uses PCA
        uses_pca = 'pca' in selected_model_file.lower()
        if uses_pca:
            st.sidebar.info("This model uses PCA for dimensionality reduction.")
    
    # Input method selection
    input_method = st.sidebar.radio("Select input method", ["Upload CSV", "Use sample data"])
    
    # Main content
    if input_method == "Upload CSV":
        st.subheader("📤 Upload your data")
        uploaded_file = st.file_uploader("Upload a CSV file with client information", type=["csv"])
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(data.head())
            except Exception as e:
                st.error(f"Error reading file: {e}")
                data = None
        else:
            data = None
    else:  # Use sample data
        st.subheader("🧪 Using sample data")
        
        # Load raw data instead of preprocessed data
        try:
            raw_path = 'data/raw/default_of_credit_card_clients.xls'
            if os.path.exists(raw_path):
                st.info("Loading raw data sample...")
                raw_data = pd.read_excel(raw_path, header=1)
                # Take a sample for demonstration
                data = raw_data.sample(n=200, random_state=42)
                # Set target values
                true_values = data['default payment next month'].values
                st.write("Preview of sample raw data:")
                st.dataframe(data.head())
                st.info(f"Sample data loaded: {data.shape[0]} records with {data.shape[1]} features.")
            else:
                st.error("Raw data not found. Please upload a file instead.")
                data = None
                true_values = None
        except Exception as e:
            st.error(f"Error loading sample data: {e}")
            data = None
            true_values = None
    
    # Make predictions when both model and data are available
    if selected_model_path and data is not None:
        st.header("🔮 Making predictions")
        
        with st.spinner("Loading model and processing data..."):
            # Load the model
            model_data = load_model(selected_model_path)
            
            if model_data is None:
                st.error("Failed to load the model.")
            else:
                # Load PCA and scaler if needed
                pca_model_path = os.path.join(model_dir, pca_file) if pca_file and 'pca' in selected_model_file.lower() else None
                scaler_model_path = os.path.join(model_dir, scaler_file) if scaler_file and 'pca' in selected_model_file.lower() else None
                
                # Apply feature engineering
                try:
                    processed_data = apply_feature_engineering(data, model_data, pca_model_path, scaler_model_path)
                    
                    if processed_data is None:
                        st.error("Error processing data. Please check your input.")
                        st.stop()
                        
                    # Preview processed data
                    st.write("Preview of processed data:")
                    st.dataframe(processed_data.head())
                    
                    # Make predictions
                    model = model_data['model']
                    model_type = type(model).__name__
                    
                    # Get probabilities if model supports it
                    if hasattr(model, 'predict_proba'):
                        y_prob = model.predict_proba(processed_data)[:, 1]
                    else:
                        y_prob = None
                    
                    # Get predictions
                    y_pred = model.predict(processed_data)
                    
                    # Display results
                    st.subheader("📊 Prediction Results")
                    
                    # Create columns for layout
                    col1, col2 = st.columns(2)
                    
                    # Show prediction distribution
                    with col1:
                        st.write("Prediction Distribution")
                        pred_df = pd.DataFrame({
                            'Default Prediction': ['No Default', 'Default'],
                            'Count': [sum(y_pred == 0), sum(y_pred == 1)]
                        })
                        st.bar_chart(pred_df.set_index('Default Prediction'))
                    
                    # Show probability distribution if available
                    if y_prob is not None:
                        with col2:
                            st.write("Default Probability Distribution")
                            fig, ax = plt.subplots()
                            sns.histplot(y_prob, bins=20, kde=True, ax=ax)
                            ax.set_xlabel('Probability of Default')
                            ax.set_ylabel('Frequency')
                            st.pyplot(fig)
                    
                    # Feature importance for tree-based models
                    if model_type in ['DecisionTreeClassifier', 'RandomForestClassifier', 'XGBClassifier']:
                        st.subheader("🔍 Feature Importance")
                        if hasattr(model, 'feature_importances_'):
                            feature_names = processed_data.columns
                            fig_importance = display_feature_importance(model, feature_names)
                            if fig_importance:
                                st.pyplot(fig_importance)
                        else:
                            st.info("Feature importance not available for this model.")
                    
                    # If we have true values (sample data), show evaluation metrics
                    if input_method == "Use sample data" and true_values is not None:
                        st.header("📏 Model Evaluation")
                        
                        # Calculate metrics
                        metrics = calculate_metrics(true_values, y_pred, y_prob)
                        
                        # Display metrics in a nice format
                        st.subheader("Performance Metrics")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Accuracy", f"{metrics['Accuracy']:.2%}")
                        col2.metric("Precision", f"{metrics['Precision']:.2%}")
                        col3.metric("Recall", f"{metrics['Recall']:.2%}")
                        col4.metric("F1 Score", f"{metrics['F1 Score']:.2%}")
                        
                        if "AUC" in metrics:
                            st.metric("AUC", f"{metrics['AUC']:.2%}")
                        
                        # Show confusion matrix
                        st.subheader("Confusion Matrix")
                        cm_plot = plot_confusion_matrix(true_values, y_pred)
                        st.pyplot(cm_plot)
                        
                        # Show ROC curve if probabilities are available
                        if y_prob is not None:
                            st.subheader("ROC Curve")
                            roc_plot, _ = plot_roc_curve(true_values, y_prob)
                            st.pyplot(roc_plot)
                            
                            st.subheader("Precision-Recall Curve")
                            pr_plot = plot_precision_recall_curve(true_values, y_prob)
                            st.pyplot(pr_plot)
                
                except Exception as e:
                    st.error(f"Error making predictions: {e}")
                    st.exception(e)
    
    # Information section
    st.sidebar.header("ℹ️ About")
    st.sidebar.info(
        """
        This app predicts credit card default using machine learning models trained on a Taiwan credit card dataset (2005).
        
        The models analyze payment history, bill amounts, and demographics to identify default risk.
        
        **Models available:**
        - Naive Bayes
        - K-Nearest Neighbors (KNN)
        - Decision Trees
        - Random Forests
        - XGBoost
        
        The app applies the same feature engineering pipeline used during model training.
        """
    )

if __name__ == "__main__":
    main()
