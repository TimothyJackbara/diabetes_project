# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 23:48:46 2023

@author: Timothy
"""

## this is to use our streamlet library in order to create the web app

import numpy as np
import pickle   # used for loading the saved model
import streamlit as st   # used for deployment of the model/creation of web page

# Loading the saved model
loaded_model = pickle.load(open("C:/Users/Timothy/Desktop/Diabetes Project/trained_model.sav", "rb"))

# creating a function for prediction

def diabetes_prediction(input_data):
    
    # changing the input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting

    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print("The outcome is",prediction)

    if (prediction[0] == 0):
        return "The patient is not diabetic."
    else:
        return "The patient is diabetic."
    


def main():
    
    # giving a title
    st.title("Timothy's Diabetes Prediction System")
    
    # getting the input data from the user
    
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure Value")
    SkinThickness = st.text_input("Skin Thickness Value")
    Insulin = st.text_input("Insulin level")
    BMI = st.text_input("BMI Value")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function Value")
    Age = st.text_input("Age")
    
    #code for prediction
    diagnosis = ""
    
    # creating a button for prediction
    
    if st.button("Diabetes Test Result"):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
    
    st.success(diagnosis)




if __name__ == "_main_":
    main()        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


    