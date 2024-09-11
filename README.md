# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:Sheetal.R
RegisterNumber: 212223230206
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
*/
```

## Output:
![Screenshot 2024-09-11 030436](https://github.com/user-attachments/assets/6ea99047-e68c-4fca-a2d1-9ecc96bcd220)


![Screenshot 2024-09-11 030550](https://github.com/user-attachments/assets/1de3094c-37d7-4637-9f13-4f162713e904)


![Screenshot 2024-09-11 030628](https://github.com/user-attachments/assets/f8304b53-f694-4042-8c47-c96b567f1f04)


![Screenshot 2024-09-11 031107](https://github.com/user-attachments/assets/0c1f0b8e-b333-4d94-83ad-b866346e6d0b)


![Screenshot 2024-09-11 031317](https://github.com/user-attachments/assets/6c956cc0-833c-469c-b47b-8c0622fd58b4)


![Screenshot 2024-09-11 031539](https://github.com/user-attachments/assets/8feb0398-d142-47e3-9657-4e77f57f4bf6)


![Screenshot 2024-09-11 031656](https://github.com/user-attachments/assets/a8f1cfd2-917d-4470-a984-5203318d43fb)


![Screenshot 2024-09-11 031800](https://github.com/user-attachments/assets/f7e0329a-13f9-4f29-9df5-61cf01fc1bdb)











## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
