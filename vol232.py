# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report,confusion_matrix

df = pd.read_csv('kc_house_data.csv')
df1 = pd.read_csv('kc_house_data.csv')
# null değer bulunmamakta
# df.isnull().sum()

# plt.figure(figsize=(21,21))
# sns.heatmap(df.corr(),annot=True)

# ysa bitirme biyo web veri tabanı 
# turkcell proje ysa web veri biyo bitirme 

# plt.figure(figsize=(10,10))
# sns.scatterplot(data=df,x='long',y='lat',hue='price',palette='RdYlGn')

df['date'] = pd.to_datetime(df['date'])

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

df = df.drop('date',axis=1)
df = df.drop('id',axis=1)

X = df.drop('price',axis=1).values
y = df['price'].values

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(19,activation='relu'))
model.add(Dense(25,activation='relu'))

model.add(Dense(25,activation='relu'))

model.add(Dense(25,activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')
model.fit(X_train,y_train,
          validation_data=(X_test,y_test),
          batch_size=128,epochs=150)
0.01

loss_df = pd.DataFrame(model.history.history)
loss_df.plot()
loss_df.plot()
from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score
preds = model.predict(X_test)

print(preds)
explained_variance_score(y_test,preds)
plt.figure(figsize=(10,10))
plt.scatter(y_test,preds)

print(explained_variance_score(y_test,preds))

