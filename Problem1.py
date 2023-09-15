'''
1. Using pickle library - save and load methods
2. Using joblib library - save and load methods
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pickle, joblib

# import the dataset
dataset = pd.read_csv('iris.csv')

X = dataset.iloc[:, : -1].values
Y = dataset.iloc[:, -1].values

# train test split
X_train, X_test, y_train, y_test = train_test_split(
	X, Y, test_size=0.2, random_state=0)

# create a linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# save the model
pkl_filename = 'pkl_linear_model.sav'
joblib_filename = 'jlib_linear_model.sav'
pickle.dump(regressor, open(pkl_filename, 'wb'))
joblib.dump(regressor,open(joblib_filename,'wb'))
# load the model
load_pkl_model = pickle.load(open(pkl_filename, 'rb'))
load_jlib_model = joblib.load(open(joblib_filename, 'rb'))

y_pred = load_pkl_model.predict(X_test)
print('root mean squared error pkl: ', np.sqrt(
	metrics.mean_squared_error(y_test, y_pred)))
y_pred = load_jlib_model.predict(X_test)
print('root mean squared error jlib : ', np.sqrt(
	metrics.mean_squared_error(y_test, y_pred)))
