#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Getting the dataset
dataset = pd.read_excel('Round 1 Estimations.xls')

# selected features for MLR = (City, SqFtTotFn, Lot-Acres, Year Built, Total Stories, Rooms, Garage Capacity)
X = dataset.iloc[:,[5,13,23,25,26,33,35]].values
y = dataset.iloc[:,2:3].values

# Removing Outliers
X = np.delete(X, (12), axis=0)
y = np.delete(y,(12),axis=0)

# Preprocessing
X[:,3] = [2017-i for i in X[:,3]]
X[:,4] = [str(i) for i in X[:,4]]
X[:,4][X[:,4]=='4+'] = '5' 
X[:,4] = [float(i) for i in X[:,4]]


#Encoding the String variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder = LabelEncoder()
X[:,0] = labelEncoder.fit_transform(X[:,0].astype(str))


#Taking care of NaNs
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy = 'mean', axis = 0)
imputer.fit(X)
X = imputer.transform(X)

#Encoding Categorical Feature 'City'
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

#Getting rid of dummy variable plot
X = X[:,1:]

#Check the ndarray
checkX = pd.DataFrame(X)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X[:,[25]] = sc_X.fit_transform(X[:,[25]])
X[:,[26]] = sc_X.fit_transform(X[:,[26]])
X[:,[27]] = sc_X.fit_transform(X[:,[27]])
X[:,[28]] = sc_X.fit_transform(X[:,[28]])
X[:,[29]] = sc_X.fit_transform(X[:,[29]])
X[:,[30]] = sc_X.fit_transform(X[:,[30]])



#Split the dataset into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


#Getting the model ready
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)


#Predict Price of flat
y_pred = regressor.predict(X_test)


#Difference in Prediction
diff = abs(y_test-y_pred)


#Number of test set apartments as a list from 1-...
listOfX = np.arange(1,1205,step=1)


# Visualising the Test set results
plt.figure(figsize=(10,5))
plt.scatter(listOfX, y_test, color = 'red')
plt.scatter(listOfX, y_pred, color = 'blue')
plt.title('Test Price (Red) vs Predicted Price (Blue)')
plt.xlabel('Apartments')
plt.ylabel('Price')
plt.show()

"""
Improvements: Features can be reduced using backward elimination later on
"""