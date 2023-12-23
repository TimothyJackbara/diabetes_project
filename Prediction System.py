# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

# Loading the saved model
loaded_model = pickle.load(open("C:/Users/Timothy/Desktop/Diabetes Project/trained_model.sav", "rb"))

input_data= (10,168,74,0,0,38,0.537,34)

# changing the input data to numpy array

input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


prediction = loaded_model.predict(input_data_reshaped)
print("The outcome is",prediction)

if (prediction[0] == 0):
    print("The patient is not diabetic.")
else:
    print("The patient is diabetic.")