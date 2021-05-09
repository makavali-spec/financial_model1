import math 
import pandas_datareader as web
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt


plt.style.use('fivethirtyeight')

#Get the data of Apple

df = web.DataReader('AAPL', data_source='yahoo', start='2017-01-01', end='2020-12-17')

# Show Data in the dataframe
df

#No. of rows and columns
df.shape

# Visualize the closing price

plt.figure(figsize=(16,8))
plt.title("Closing Price History")
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD($)', fontsize=18)
plt.show()

#Create new dataframe with only close column

data = df.filter(['Close'])

#Convert the dataframe to a numpy array

dataset = data.values

#Get the number of rows to train the model on

training_data = len(dataset) * 0.8

#Round off the dataset
training_data_len = math.ceil(training_data)

training_data_len

#Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

scaled_data

#Create the training dataset

train_data = scaled_data[0:training_data_len: 1]
#Split the data into X_train, y_train

x_train = []
y_train = []

for i in range(60, len(train_data)):
	x_train.append(train_data[i-60:i, 0])
	y_train.append(train_data[i,0])

	if i <= 60:
		print(x_train)
		print(y_train)
		print()

#Conver the x_train. y_train into numpy array

x_train, y_train = np.array(x_train), np.array(y_train)

#Reshape the data
x_train.shape

x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1], 1))
x_train.shape


#Build theLSTM model

model = Sequential()
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))


#Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics='accuracy')

#Train the model
model.fit(x_train, y_train, batch_size=1, epochs=5)

#Create the testing data set
#Create a new array containing scaled value

test_data = scaled_data[training_data_len-60:, : ]

#Create the datasets x_test, y_test
x_test = []
y_test = dataset[training_data_len:, :]

for i in range(60, len(test_data)):
	x_test.append(test_data[i-60:i, 0])


#Convert the data to a numpy array

x_test = np.array(x_test)

#Shape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#Get the model predicted price values

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#Get the root mean squared error(RMSE)
rmse = np.sqrt(np.mean(predictions - y_test)**2)
print(rmse)

#Round off value
print(round(rmse, 6))

train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

plt.figure(figsize=(16,8))
plt.title('Prediction Model on Stock Analysis')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Actual_Val', 'Predictions'], loc='lower right')
plt.show()
