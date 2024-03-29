# -*- coding: utf-8 -*-
"""Diabetes_Prediction

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11nbON0oiYCzsF9m1XTHIg0aDeJItSvTh

# Importing the Dependencies
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

"""# Data Collection and Analysis

PIMA Dataset from Kaggle
"""

diabetes_dataset = pd.read_csv('/diabetes.csv')

# Printing the first 5 rows of the dataset
diabetes_dataset.head()

# number of rows and colums in the dataset
diabetes_dataset.shape

#Getting the statistical measures of the data
diabetes_dataset.describe()

diabetes_dataset['Outcome'].value_counts()

"""0 --> Non-Diabetic
1 --> Diabetic
"""

diabetes_dataset.groupby('Outcome').mean()

# Seperating the data and labels

X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']

print(X)

print(Y)

"""Data Standardization"""

scaler = StandardScaler()

scaler.fit(X)

standardized_data = scaler.transform(X)

print(standardized_data)

X = standardized_data
Y = diabetes_dataset['Outcome']

print(X)

print(Y)

"""Train Test Split"""

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size  = 0.3, stratify=Y, random_state = 2 )

print(X.shape, X_train.shape, X_test.shape)

"""Model Training"""

classifier = svm.SVC(kernel = 'linear')

#training the Support Vector Machine
classifier.fit(X_train, Y_train)

"""Model Evaluation

Accuracy Score
"""

#Accuracy score on the Training data
X_train_prediction  = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data: ', training_data_accuracy)

#Accuracy score on the Test data
X_test_prediction  = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the training data: ', test_data_accuracy)

"""Making a Predictive system"""

input_data = (5,116,74,0,0,25.6,0.201,30)

#changing the input_data to numpy array
input_dat_as_numpy_array = np.asarray(input_data)

#reshape the data as we are prediciting for one instance
input_data_reshaped = input_dat_as_numpy_array.reshape(1, -1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] ==0 ):
  print('The person is not diabetic')
else:
  print('The person is diabetic')

"""**Saving** the Trained Model"""

import pickle

filename = 'trained_model.sav'
pickle.dump( classifier, open(filename, 'wb'))

#loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

input_data = (5,116,74,0,0,25.6,0.201,30)

#changing the input_data to numpy array
input_dat_as_numpy_array = np.asarray(input_data)

#reshape the data as we are prediciting for one instance
input_data_reshaped = input_dat_as_numpy_array.reshape(1, -1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = loaded_model.predict(std_data)
print(prediction)

if (prediction[0] ==0 ):
  print('The person is not diabetic')
else:
  print('The person is diabetic')

