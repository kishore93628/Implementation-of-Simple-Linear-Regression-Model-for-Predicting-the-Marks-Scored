# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.import the required libraries and read the dataframe.

2.Assign hours to X and scores to Y.

3.Implement training set and test set of the dataframe.

4.Plot the required graph both for test data and training data.

5.Find the values of MSE , MAE and RMSE.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: kishore
RegisterNumber: 212221080042
*/
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: NARENDRAN B
RegisterNumber: 212222240069

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df = pd.read_csv('dataset/student_scores.csv')
df.head()

#segregating data to variables
x = df.iloc[:, :-1].values
x

#splitting train and test data
y = df.iloc[:, -1].values
y

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression 
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

#displaying predicted values
y_pred

#displaying actual values
y_test

#graph plot for training data
plt.scatter(x_train,y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
plt.scatter(x_test,y_test,color="purple")
plt.plot(x_train,regressor.predict(x_train),color="green")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print("MSE= ",mse)

mae=mean_absolute_error(y_test,y_pred)
print("MAE = ",mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
*/

## Output:
![image](https://github.com/kishore93628/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/164981341/7e23550f-e9fb-43d0-8e8f-daa72a43a733)
![image](https://github.com/kishore93628/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/164981341/3fa91fef-7daf-4c01-8d1b-31ab8f5e3e35)
![image](https://github.com/kishore93628/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/164981341/e0cccf64-54e3-44c8-ad18-94d7d7a9d4a8)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
