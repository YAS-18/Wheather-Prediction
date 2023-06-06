# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 20:14:34 2023

@author: ysalu
"""

import numpy as np
import pickle
import streamlit as st

st.markdown(
    """
    <style>
    body {
        background-: yellow;
    }
    </style>
    """,
    unsafe_allow_html=True
)


#loading the model
loaded_model = pickle.load(open('D:/ML/Project/svr_model.sav','rb'))


#creating the function
def temperature_predict(input_data):
    #creating the array of the input data
    data = np.asarray(input_data).reshape(1,-1)
    #finding the predicted value  
    predict_value = loaded_model.predict(data)
    #return predicted value
    return predict_value


def main():
    #Title of the Website
    st.title("Weather Prediction Web App")
    
    #awnd	prcp	snow	snwd	tmax	tmin
    awnd = st.number_input(label = 'Average Wind Speed') 
    prcp = st.number_input(label = 'Precipitation') 
    snow = st.number_input(label = 'Snow')
    snwd = st.number_input(label = 'Snow Depth') 
    tmax = st.number_input(label = 'Maximun Temperatue')
    tmin = st.number_input(label = 'Minimum Temperature')
    
    #code for prediction
    if st.button('Predict'):
        result = temperature_predict([awnd,prcp,snow,snwd,tmax,tmin])
        st.write("Tommorows Temperature : ",result[0])
        
    
    
if __name__ == '__main__':
    main()
    
    