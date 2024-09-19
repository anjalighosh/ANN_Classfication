import streamlit as st 
import numpy as np
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler, LabelEncoder,OneHotEncoder
import pandas as pd 
import pickle

#load all trained model,scaler pickle,onehot pkl also
model= tf.keras.models.load_model('model.h5')

#load encoders and scaler
with open('geo.pkl','rb') as file:
    geo_encoder=pickle.load(file)
with open('label_encoder_gender.pkl','rb') as file:
    gender_encoder=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler_pickle=pickle.load(file)

#streamlit app
st.title('Customer Churn Prediction')

#User input
geography=st.selectbox('Geography',geo_encoder.categories_[0])
gender=st.selectbox('Gender',gender_encoder.classes_)
age=st.slider('Age', 18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure', 0, 10)
num_of_products=st.slider('Number of products', 1, 4)
has_cr_card=st.selectbox('Has Credit Card', [0,1])
is_active_member=st.selectbox('Is Active Member', [0,1])

#prepare input data
input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[gender_encoder.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
    
})

#one hot encoding geography

geo_encoded= geo_encoder.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=geo_encoder.get_feature_names_out(['Geography']))
geo_encoded_df

input_data=pd.concat([input_data.reset_index(drop=True), geo_encoded_df],axis=1)

#scale data
input_data_scaled=scaler_pickle.transform(input_data)

#predict data
predict=model.predict(input_data_scaled)
predict_prob=predict[0][0]
st.write(f'Churn Probability :{predict_prob:.2f}')


if predict >0.5:
    st.write("Customer is likely to churn")
else:
    st.write("Customer is not likely to churn")

