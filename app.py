#!/usr/bin/env python
# coding: utf-8

# In[14]:


import streamlit as st
import pandas as pd
import joblib

# Load the saved model
model = joblib.load('rf_model.sav')

# Define the prediction function
def predict_cluster(features):
    # Preprocess the input features if necessary
    # Make a prediction using the loaded model
    prediction = model.predict(features)
    # Return the prediction
    return prediction

# Define the Streamlit app
def app():
    # Set the title and sidebar
    st.set_page_config(page_title='Customer Segmentation', layout='wide')
    st.sidebar.title('Customer Segmentation')
    st.sidebar.text('Enter values for each feature to predict segment.')
    
    # Load the dataset with the feature descriptions
    feature_desc = pd.read_csv('customer_segment.csv')
    
    # Define the input fields
    features = {}
    for feature in feature_desc.columns[:-1]:
        features[feature] = st.number_input(feature, value=feature_desc[feature].mean())
    
    # Define the "Predict" button
    if st.button('Predict'):
        # Store the input features in a DataFrame
        features_df = pd.DataFrame(features, index=[0])
        # Call the prediction function and store the result
        result = predict_cluster(features_df)
        # Display the predicted cluster on the app
        st.write(f'The predicted segment is: {result[0]}')

# Run the app
if __name__ == '__main__':
    app()

