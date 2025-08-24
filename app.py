import streamlit as st
import pandas as pd
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.externals import joblib



def main():
    st.title("Credit Card Default Prediction")
    st.write("This is a simple app to predict credit card default.")

    # File uploader for user to upload their data
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        # Read the data
        data = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(data)

        # Apply feature engineering
        model_data = load_model(selected_model_path)
        engineered_data = apply_feature_engineering(data, model_data)
        if engineered_data is not None:
            st.write("Engineered Data Preview:")
            st.dataframe(engineered_data)

            # TODO: Add model prediction code here


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

def apply_feature_engineering(data, model_data):
    """Apply the same feature engineering used during training"""
    try:
        # Step 1: Preprocess the data
        preprocessed_data = preprocess_data(data)
        
        # Step 2: Create engineered features
        data_dict = create_new_features(preprocessed_data)
        data_encoded = data_dict['encoded']
        
        # Step 3: Ensure correct column types
        data_encoded = set_types_encoded(data_encoded)
        
        # Step 4: Check if we have PCA transformer
        if 'pca' in model_data and 'scaler' in model_data:
            # Get features excluding target/ID
            feature_cols = [col for col in data_encoded.columns if col not in ['ID', 'default payment next month']]
            X = data_encoded[feature_cols]
            
            # Scale the features using the saved scaler
            X_scaled = model_data['scaler'].transform(X)
            
            # Apply PCA using the saved transformer
            X_pca = model_data['pca'].transform(X_scaled)
            
            # Create DataFrame with PCA components
            pca_cols = [f'PC{i+1}' for i in range(X_pca.shape[1])]
            return pd.DataFrame(X_pca, columns=pca_cols)
        
        return data_encoded
    except Exception as e:
        st.error(f"Error in feature engineering: {e}")
        st.exception(e)
        return None
