kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import csv
import re

from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam


def data_subset(data):
    features=['date','bedrooms','bathrooms','floors','sqft_living','sqft_lot','view','condition']
    lenght_features=len(features)
    subset=data[features]
    return subset,lenght_features

def create_model(train_set_size,input_lenght, num_epochs,batch_size):
    model=Sequential()
    model.add(Dense(7,input_dim=input_lenght,activation='softplus'))
    model.add(Dense(3,activation='softplus'))
    model.add(Dense(1,activation='softplus'))
    
    lr=0.01
    adam0=Adam(lr=lr)
    
    model.compile(loss='binary_crossentropy',optimizer=adam0,metrics=['accuracy'])
    
    filepath='weight.best.hdf5'
    checkpoint=ModelCheckpoint(filepath,monitor='acc',verbose=1,save_best_only=True,mode=max)
    callbacks_list=[checkpoint]
    history_model=model.fit(X_train[:train_set_size], Y_train[:train_set_size],callbacks=callbacks_list,epochs=num_epochs,batch_size=batch_size,verbose=0)
    return model,history_model

def plot(history):
    loss_history=history.history['loss']
    acc_history=history.history['acc']
    epochs=[(i+1) for i in range(num_epochs)]
    ax=plt.suplot(211)
    ax.plot(epochs,loss_history,color='red')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Error Rate per Epoch\n')
    
    ax2=plt.suplot(212)
    ax2.plot(epochs,acc_history ,color='blue')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy per Epoch\n')
    plt.subplot_adjust(hspace=0.8)
    plt.savefig('Accuracy_Loss.png')
    plt.close
    

def test(batch_size):
    test = pd.read_csv('kc_house_data.csv', header=0)
    test_ids = test['id']
    testdata, _ = data_subset(test)

    X_test = np.array(testdata).astype(float)

    output = model.predict(X_test, batch_size=batch_size, verbose=0)
    output = output.reshape((21596,))

    # Sonuçları ondalık sayı yerine 0-1 olarak değiştirebilirsiniz
    #outputBin = np.zeros(0)
    #for element in output:
    #    if element <= .5:
    #         outputBin = np.append(outputBin, 0)
    #    else:
    #        outputBin = np.append(outputBin, 1)
    #output = np.array(outputBin).astype(int)

    column_1 = np.concatenate((['id'], test_ids ), axis=0 )
    column_2 = np.concatenate( ( ['price'], output ), axis=0 )

    f = open("output.csv", "w")
    writer = csv.writer(f)
    for i in range(len(column_1)):
        writer.writerow( [column_1[i]] + [column_2[i]])
    f.close()
    
seed = 7
np.random.seed(seed)


# veri yukleme
veriler = pd.read_csv('kc_house_data.csv') #train

veriler.info()
id ve price sütunu ayrılıp dataset eğitime hazır hale getirilmiştir
df_price=veriler.iloc[:,2:3]
df_date=veriler.iloc[:,1:2]
df_other_columns=veriler.iloc[:,3:]
df_train=pd.concat([df_date,df_other_columns],axis=1) #preprocess
Price_values=df_price.values
train_values=df_train.values
date sütununu ayı yıl gün olarak bölme
veriler['date']=pd.to_datetime(veriler['date'])
veriler['month']=veriler['date'].dt.month
veriler['year']=veriler['date'].dt.year
veriler['day']=veriler['date'].dt.day
deneme=veriler.drop(columns=['date','id'])
veriler['month'] = data['Date.of.Birth'].dt.month
data[['Date.of.Birth','month']].head()
data['day'] = data['Date.of.Birth'].dt.day
data[['Date.of.Birth','day']].head()
data['year'] = data['Date.of.Birth'].dt.year
data[['Date.of.Birth','year']].head()


num_epochs = 100
batch_size = 32 # her itesrasyonda alacağı küme sayısı



traindata, lengh_features = data_subset(veriler)

Y_train = np.array(veriler['price']).astype(int)
X_train = np.array(traindata).astype(float)


train_set_size = int(.67 * len(X_train))


model, history_model = create_model(train_set_size, lengh_features, num_epochs, batch_size)

plot(history_model)


X_validation = X_train[train_set_size:]
Y_validation = Y_train[train_set_size:]


loss_and_metrics = model.evaluate(X_validation, Y_validation, batch_size=batch_size)
print ("loss_and_metrics")

test(batch_size)