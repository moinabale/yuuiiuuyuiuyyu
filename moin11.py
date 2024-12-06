import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
df=pd.DataFrame({'experience':[1,2,3,4,5,6,7,8,9,10],'salary':[10000,20000,30000,40000,50000,60000,70000,80000,90000,None]})
print("checking missing value",df.isnull().sum())
data=df['salary'].mean()
df['salary']=df['salary'].fillna(data)
print(df.isnull().sum())
x=[['experience']]
y=['salary']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression().fit(x_train,y_train)


