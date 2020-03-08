#!/usr/bin/python36
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error

print("content-type: text/html\n")
print(" <h4> Estimating Consultation Time ....... </h4> </br>")

import pandas as pd
data = pd.read_csv('data4.csv')
y = data['Consultation Time ']
x = data.iloc[:, :6] 
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x,y)
y=model.predict([[2,e,g,l,2,4]])
print(" <h2> The Estimated time is {} </h2>".format(c))
wt = linreg.predict(np.array([f,1,4,1,0,0,0,0]).reshape(1,-1))[0]
print("<h3> The Average Waiting Time is : {} </h3>".format(wt))
