## Import all the Libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st
import pickle

## Load all the models
### Load OHE, Label_encoder and Scaler pickle file
with open('ohe_geography.pkl', 'rb') as file:
    ohe_geography = pickle.load(file)
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
### Load ANN Model pickle
ann_model = tf.keras.models.load_model('ann_model.h5')

## Streamlit app
st.title('Customer Churn Prediction')

## User Input
geography = st.selectbox('Geography', ohe_geography.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credict Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

## Prepare the input data
input_data = pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})

### OHE
geo = ohe_geography.transform([[geography]])
geo_df = pd.DataFrame(geo, columns = ohe_geography.get_feature_names_out())

input_df = pd.DataFrame(input_data)

input_df = pd.concat([geo_df, input_df], axis = 1)
input_scale = scaler.transform(input_df)

## Predict Churn
prediction = ann_model.predict(input_scale)
predict_proba = prediction[0][0]

st.write("Probability: ", predict_proba)

if predict_proba > 0.5:
    st.write('The Customer is likely to Churn.')
else:
    st.write('The customer is not likely to churn.')