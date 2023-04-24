#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 19:22:51 2023

@author: sparsh
"""

import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('/Users/sparsh/Downloads/trained_model.sav', 'rb'))

#creating a fuction

def diabetes_function(input_data):
    
    
    
    #changing the input_data to numpy array
    input_dat_as_numpy_array = np.asarray(input_data)

    #reshape the data as we are prediciting for one instance
    input_data_reshaped = input_dat_as_numpy_array.reshape(1, -1)


    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] ==0 ):
      return'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
    
def main():
    
    
    
    
    
    
    # Giving a title
    st.title('Diabetes Prediction Web App')
    
    # Getting the input Data from the user

    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('BloodPressure Level')
    SkinThickness = st.text_input('SkinThickness Value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI Value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    Age = st.text_input('Age of the Person')
    
    
    
    #code for prediction
    diagnosis = ''
    
    # Creating a button for prediction
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_function([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
    st.success(diagnosis)
    
    
    
if __name__ == '__main__':
    main()

    
    
    
    