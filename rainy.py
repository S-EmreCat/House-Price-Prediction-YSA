# -*- coding: utf-8 -*-
"""rainy.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/110A_4KBWvQLWHQBB8zxfrm2nHx-Mys8Z
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error,explained_variance_score
from keras.layers.core import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.layers.core import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.datasets import mnist
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from keras.layers.core import Activation
from sklearn import preprocessing
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
import seaborn as sns
plt.style.use("fivethirtyeight")
sns.set_style('whitegrid')
# %matplotlib inline

df=pd.read_csv('csgo_round_snapshots.csv')

df.head()

df['round_winner'].replace('CT',1,inplace=True)
df['round_winner'].replace('T',0,inplace=True)
df['bomb_planted'].replace(True,1,inplace=True)
df['bomb_planted'].replace(False,0,inplace=True)
map=df.iloc[:,3:4].values
le=preprocessing.LabelEncoder()
map[:,0]=le.fit_transform(df.iloc[:,3:4])
ohe=preprocessing.OneHotEncoder()
map=ohe.fit_transform(map).toarray()
sonuc= pd.DataFrame(data=map,columns=['de_dust2','de_mirage','de_nuke','de_inferno','de_overpass','de_vertigo','de_train','de_cache'])
sonuc.head()
df=df.drop(['map'],axis=1)
df=pd.concat([sonuc,df],axis=1)
df.head()
#df.map.unique()

X= df.drop('round_winner',axis=1)
y=df.round_winner.values.reshape(-1,1)
X.shape

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=1)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = Sequential()


model.add(Dense(100, input_dim=X_train.shape[1], activation='elu'))
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(1))
model.summary()
metrics = ['accuracy']
optimizer = keras.optimizers.Adam(lr=0.01)
model.compile(optimizer=optimizer, loss='mean_squared_error',metrics=metrics)
model.optimizer.get_config()

model.summary()
num_epochs=50
history=model.fit(X_train,y_train, validation_data=(X_test,y_test),batch_size=128,epochs=num_epochs,validation_split=0.20)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss Progression During Training/Validation')
plt.ylabel('Training and Validation Losses')
plt.xlabel('Epoch Number')
plt.legend(['Training Loss', 'Validation Loss'])

# Plot the accuracy curves for training and validation.



from sklearn.metrics import accuracy_score
y_pred1 = model.predict(X_test)
y_pred = np.argmax(y_pred1, axis=1)

# Printaccuracy score
print(accuracy_score(y_test, y_pred)*100)

def plotLoss(history):  
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.figure(figsize=(20,20))
    plt.show()
    
plotLoss(history)