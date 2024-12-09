import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

data = pd.read_csv('simulated_data.csv')

T = np.array(data['T'])
S1 = np.array(data['IVIG']).reshape((-1, 1))
S2 = np.array(data['SCIG']).reshape((-1, 1))

split = 400

T_train = T[:split]
T_test = T[split:]

S1_train = S1[:split]
S1_test = S1[split:]

S2_train = S2[:split]
S2_test = S2[split:]

memory = 5
batch_size = 20
num_epochs = 25

train_gen1 = TimeseriesGenerator(S1_train, S1_train, length=memory, batch_size=batch_size)
test_gen1 = TimeseriesGenerator(S1_test, S1_test, length=memory, batch_size=1)

train_gen2 = TimeseriesGenerator(S2_train, S2_train, length=memory, batch_size=batch_size)
test_gen2 = TimeseriesGenerator(S2_test, S2_test, length=memory, batch_size=1)

model1 = Sequential()
model1.add(LSTM(10, activation='relu', input_shape=(memory, 1)))
model1.add(Dense(1))
model1.compile(optimizer='adam', loss='mse')

model2 = Sequential()
model2.add(LSTM(10, activation='relu', input_shape=(memory, 1)))
model2.add(Dense(1))
model2.compile(optimizer='adam', loss='mse')

model1.fit(train_gen1, epochs=num_epochs, verbose=1)

model2.fit(train_gen2, epochs=num_epochs, verbose=1)

S1_pred = model1.predict(test_gen1)

S2_pred = model2.predict(test_gen2)

T_train = T_train.reshape((-1))
T_test = T_test.reshape((-1))

S1_train = S1_train.reshape((-1))
S2_train = S2_train.reshape((-1))

S1_test = S1_test.reshape((-1))
S2_test = S2_test.reshape((-1))

S1_pred = S1_pred.reshape((-1))
S2_pred = S2_pred.reshape((-1))

plt.plot(T, S1, 'r', label='IVIG')
plt.plot(T_test[:len(S1_pred)], S1_pred, 'b', label='IVIG Prediction')
plt.plot(T, S2, 'g', label='SCIG')
plt.plot(T_test[:len(S1_pred)], S2_pred, 'y', label='SCIG Prediction')
plt.title('LSTM Predictions')
plt.xlabel('Time')
plt.ylabel('Demand')
plt.legend()
plt.show()

mse1 = np.mean((S1_test - S1_pred[:len(S1_test)]) ** 2)
mse2 = np.mean((S2_test - S2_pred[:len(S2_test)]) ** 2)

print('IVIG MSE:', mse1)
print('SCIG MSE:', mse2)