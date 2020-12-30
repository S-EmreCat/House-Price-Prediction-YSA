# kutuphaneler

# -*- coding: utf-8 -*-
#libraries
import pandas as pd
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error,explained_variance_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns


df = pd.read_csv('kc_house_data.csv')
# null değer bulunmamakta
df.isnull().sum()


df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df = df.drop('date',axis=1)
df = df.drop('id',axis=1)
# print(df.head(10))
# print(df.dtypes)
df_othercolumns=df.drop('price',axis=1)
df_price=df['price']

X = df.drop('price',axis=1).values
Y = df_price.values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=101)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# KORELASYON MATRİSİ RENKLERLE
def plot_corr(df):
    f, ax = plt.subplots(figsize=(10, 10))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(df.corr(), cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('Korelasyon Matrisi')    
plot_corr(df)
# KORELASYON MATRİSİ SAYILARLA
# plt.figure(figsize=(15,15))
# sns.heatmap(df.corr(),annot=True)
model = Sequential()
model.add(Dense(19,input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(1))
# lr belirleme
lr=0.01
adam0=Adam(lr=lr)
#loss değeri mse olarak belirlendi
# adam optimizasyonu ile hata minimize edilmeye çalışıldı
# metrics olarak              tercih edildi
model.compile(optimizer=adam0,loss='mse')
num_epochs = 50
history=model.fit(X_train,Y_train, validation_data=(X_test,Y_test),
          batch_size=64,epochs=num_epochs,validation_split=0.2)


loss_df = pd.DataFrame(model.history.history)
plt.title('')
loss_df.plot()
preds = model.predict(X_test)
# varyans puanı
print('Variance Score: ',explained_variance_score(Y_test,preds))
print('Mean Absolute Error: ',mean_absolute_error(Y_test, preds))

