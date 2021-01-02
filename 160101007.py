# kutuphaneler

# -*- coding: utf-8 -*-
#libraries
import pandas as pd
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error,explained_variance_score,r2_score,mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

df = pd.read_csv('kc_house_data.csv')  # model oluşturulurken kullanılacak df
df1 = pd.read_csv('kc_house_data.csv') # gerçek df yi görmek için değişiklik yapılmamış hali
# null değer bulunmamakta
df.isnull().sum()
# date sütunu model için uygun hale getirildi.
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
# drop columns
df = df.drop('date',axis=1) 
df = df.drop('id',axis=1)
# df hakkında genel bilgi
print(df.head(10))
print(df.dtypes)
df_othercolumns=df.drop('price',axis=1)
df_price=df['price']
X = df.drop('price',axis=1).values
Y = df_price.values
# veriyi eğitim ve test olarak bölme 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train.shape[1])

model = Sequential()
model.add(Dense(20,input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(1))
# lr,epoch sayısı belirleme
lr=0.07
numberofepochs = 300
# adam optimizasyonu kullanıldı 
optimizer_Adam=Adam(lr=lr)
batch_size=256
# loss değeri mean_absolute_error olarak belirlendi
model.compile(optimizer=optimizer_Adam,loss='mean_absolute_error')
history=model.fit(X_train,Y_train, validation_data=(X_test,Y_test),batch_size=batch_size,epochs=numberofepochs,validation_split=0.2)
Y_prediction = model.predict(X_test)
# korelasyon matrisi çizimi
def plot_Corr(df):
    f, ax = plt.subplots(figsize=(10, 10))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(df.corr(), cmap=cmap, vmax=1.0, vmin=-1.0, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('Correlation Matrix')    
    plt.show()
    plt.close()
# plot_Corr(df)
# korelasyon matrisinin sayılar ile gösterimi
# plt.figure(figsize=(15,15))
# sns.heatmap(df.corr(),annot=True)
# Loss değerlerine göre grafik
def plot_Loss(history):  
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Train_Loss & Validation_Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train_Loss', 'Validation'], loc='upper right')
    plt.show()
    plt.close()
plot_Loss(history)

# prediction vs y_test
def plot_Count(history):  
    plt.plot(Y_test, color = 'yellow', label = 'Real data')
    plt.plot(Y_prediction, color = 'red', label = 'Predicted data')
    plt.title('Prediction & True')
    plt.legend()
    plt.show()
    plt.close()
plot_Count(history)

print('Artifical Neural Network:')
print('Mean Absolute Error:', mean_absolute_error(Y_test,Y_prediction))
print('Mean Squared Error:', mean_squared_error(Y_test,Y_prediction))
print('Explained Variance Score:', explained_variance_score(Y_test,Y_prediction))
print('R2 Score:', r2_score(Y_test,Y_prediction))

