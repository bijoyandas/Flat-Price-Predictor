#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Getting the dataset
dataset = pd.read_excel('Round 1 Estimations.xls')
X = dataset.iloc[:,13:14].values
y = dataset.iloc[:,2:3].values

#Getting rid of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy = 'mean', axis = 0)
imputer.fit(X)
X = imputer.transform(X)

#Split the dataset into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=0)

#Getting the model ready
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predict Price of flat
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Price vs Sq.Fit (Training Set)')
plt.xlabel('Square Fit')
plt.ylabel('Price')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Price vs Sq.Fit (Test set)')
plt.xlabel('Square Fit')
plt.ylabel('Price')
plt.show()