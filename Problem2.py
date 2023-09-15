import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

dataset = pd.read_csv('Churn_Modelling.csv')

# 1. Information about the Data
print(dataset.head())
print(dataset.info())

# 2. Exploratory Data Analysis and Visualization
plt.figure(figsize=(12,6))
sns.heatmap(dataset.corr(),
			cmap='BrBG',
			fmt='.2f',
			linewidths=2,
			annot=True)
# plt.show()

lis = ['SeniorCitizen', 'tenure', 'MonthlyCharges']
plt.subplots(figsize=(3, 1))
index = 1
for i in lis:
	plt.subplot(2, 2, index)
	sns.distplot(dataset[i])
	index += 1
# plt.show()

lis2 = ['Partner', 'DeviceProtection']
plt.subplots(figsize=(10, 5))
index = 1
for col in lis2:
	y = dataset[col].value_counts()
	plt.subplot(1, 2, index)
	plt.xticks(rotation=90)
	sns.barplot(x=list(y.index), y=y)
	index += 1
# plt.show()

# 3. Data Preprocessing

# Check if any data has any null entries. -> boolean array
print(dataset.isnull().any())

# We dont have any nulls in this case but to fill any column.
# dataset["PhoneService"].fillna(dataset["PhoneService"].mode()[0],inplace = True)
# dataset["gender"].fillna(dataset["gender"].mode()[0],inplace = True)
# dataset["StreamingTV"].fillna(dataset["StreamingTV"].mean(),inplace = True)

# 4. Label Encoding -> Text to numerical
le = LabelEncoder()
dataset['gender'] = le.fit_transform(dataset["gender"])
dataset['PaperlessBilling'] = le.fit_transform(dataset["PaperlessBilling"])
dataset['SeniorCitizen'] = le.fit_transform(dataset["SeniorCitizen"])
dataset['Partner'] = le.fit_transform(dataset["Partner"])
dataset['Dependents'] = le.fit_transform(dataset["Dependents"])
dataset['tenure'] = le.fit_transform(dataset["tenure"])
dataset['Churn'] = le.fit_transform(dataset['Churn'])
dataset['PhoneService'] = le.fit_transform(dataset['PhoneService'])
dataset['Contract'] = le.fit_transform(dataset['Contract'])
print(dataset.head())

# 5. Splitting Dependent and Independent Variables
x = dataset.iloc[:,1:5].values
y = dataset.iloc[:,-1].values

# Splitting into test and train data 20-80 split
x_train, x_test, y_train, y_test = train_test_split(x,y,
													test_size = 0.2,
													random_state = 0)

# 5. Feature Scaling -> Normalize the features
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


# Training 4 models -> KNN, RF, SVC, Logistic Regression
knn = KNeighborsClassifier(n_neighbors=3)
rfc = RandomForestClassifier(n_estimators = 7,
							criterion = 'entropy',
							random_state =7)
svc = SVC()
lc = LogisticRegression()

for clf in (rfc, knn, svc,lc):
	clf.fit(x_train, y_train)
	y_pred = clf.predict(x_test)
	print("Accuracy score of ",clf.__class__.__name__,"=",
		100*metrics.accuracy_score(y_test, y_pred))
