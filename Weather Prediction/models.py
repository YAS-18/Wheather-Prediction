# -*- coding: utf-8 -*-
"""models.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YVlhyUOUetm61y3sJkb7S03m4j98czlZ
"""

!pip install streamlit -q

!pip install pickle

# import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import xgboost as xgb
import streamlit as st
import pandas as pd
import numpy as np

# load data into a Pandas DataFrame
data = pd.read_csv('weather.csv')
data.sample(5)
# set the date column as the index
data.set_index('date', inplace=True)

# split the data into features and target variable
X = data.drop(['target','station','name'], axis=1)
y = data['target']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create a Random Forest model and fit it to the training data
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# evaluate the model on the testing data
y_pred = rf_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error:', mae)

# make a prediction for the next day's temperature
last_day_data = X.iloc[-1, :]
next_day_data = last_day_data.to_frame().T
next_day_temperature = rf_model.predict(next_day_data)
print('Predicted Next Day Temperature:', next_day_temperature)
acc = rf_model.score(X_test,y_test)
print('Accuracy : ',acc)

# Create SVR model with RBF kernel
svr_model = SVR(kernel='rbf', gamma='scale', C=1.0, epsilon=0.1)

# Fit the model on the training data
svr_model.fit(X_train, y_train)

# Predict temperature on the test data
y_pred = svr_model.predict(X_test)

# Calculate the mean squared error (MSE) of the predictions
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error: ", mae)

# make a prediction for the next day's temperature
last_day_data = X.iloc[-1, :]
next_day_data = last_day_data.to_frame().T
next_day_temperature = svr_model.predict(next_day_data)
print('Predicted Next Day Temperature:', next_day_temperature)

acc = svr_model.score(X_test,y_test)
print('Accuracy : ',acc)

# Create linear regression model
lr_model = LinearRegression()

# Train the model
lr_model.fit(X_train, y_train)

# Predict weather for a new set of data

prediction = lr_model.predict(X_test)

# Calculate the mean squared error (MSE) of the predictions
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error: ", mae)

# make a prediction for the next day's temperature
last_day_data = X.iloc[-1, :]
next_day_data = last_day_data.to_frame().T
next_day_temperature = lr_model.predict(next_day_data)
print('Predicted Next Day Temperature:', next_day_temperature)

#train dataset Accuracy
train_acc = lr_model.score(X_train,y_train)
print('Accuracy : ',train_acc)

#test dataset Accuracy
test_acc = lr_model.score(X_test,y_test)
print('Accuracy : ',test_acc)

input_data = (10.51,0.04,0,0,40,32)

data = np.asarray(input_data).reshape(1,-1)

prediction = svr_model.predict(data)

print(prediction)

import pickle
filename = 'svr_model.sav'
pickle.dump(svr_model,open(filename,'wb'))

load_model = pickle.load(open('svr_model.sav','rb'))