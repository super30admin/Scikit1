'''
Get a regression model summary from sci-kit learn
1. Using sklearn package -> model.xxx attributes
2. Using stats models package formula.api

Important Points:
R-squared value: 
1. The R-squared value ranges from 0 to 1. 
2. An R-squared of 100% indicates that changes in the independent variable completely explain all changes in the dependent variable (s). 
3. If the r-squared value is 1, it indicates a perfect fit. The r-squared value in our example is 0.638.
F-statistic: 
1. The F-statistic compares the combined effect of all variables. Simply put, if your alpha level is greater than your p-value, you should reject the null hypothesis
'''
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import statsmodels.formula.api as smf
import pandas as pd

# Load the data
irisData = load_iris()

# Create feature and target arrays
X = irisData.data
y = irisData.target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

# predicting on the X_test data set
print(model.predict(X_test))

# summary of the model
print('model intercept :', model.intercept_)
print('model coefficients : ', model.coef_)
print('Model score : ', model.score(X, y))


# 2. Using statsmodels package

# loading the csv file
df = pd.read_csv('headbrain1.csv')
print(df.head())

# fitting the model
df.columns = ['Head_size', 'Brain_weight']
model = smf.ols(formula='Head_size ~ Brain_weight',
				data=df).fit()

# model summary
print(model.summary())

# In this case If we use 0.05 as our significance level, 
# we reject the null hypothesis and accept the alternative hypothesis as p< 0.05. 
# As a result, we can conclude that there is a relation between head size and brain weight.