import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
data = pd.read_csv('Boston.csv')
X1 = data.iloc[:,1:5]
X2 = data.iloc[:,6:14]
X = pd.concat([X1,X2],axis=1)
y = pd.DataFrame(data.iloc[:,14])

'''
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
'''
#X = (X - X.mean(axis=0)) / X.std(axis=0)
X = preprocessing.normalize(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1, random_state=42)

import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

classifier.add(Dense(512, activation = 'relu', input_dim = 12))
classifier.add(Dense(128, activation = 'relu'))
classifier.add(Dense(1, activation = 'relu'))

classifier.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# f=open("temp.txt","r")
# lines = f.readlines()
# df=pd.DataFrame(columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV'])
# line = []
# for i in range(len(lines)):
#     if i%2 == 0:
#         line = []
#     line_list = lines[i].replace("\n","").split(" ")
#     for j in line_list:
#         if j:
#             line.append(j)
#     if i%2 != 0:
#         df.loc[len(df)]=line
    
# df.to_csv("Boston.csv",index=True)
