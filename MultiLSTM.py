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

train_gen = TimeseriesGenerator(np.column_stack((S1_train, S2_train)), np.column_stack((S1_train, S2_train)), length=memory, batch_size=batch_size)
test_gen = TimeseriesGenerator(np.column_stack((S1_test, S2_test)), np.column_stack((S1_test, S2_test)), length=memory, batch_size=1)

model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(memory, 2)))
model.add(Dense(2))
model.compile(optimizer='adam', loss='mse')

model.fit(train_gen, epochs=num_epochs, verbose=1)

S_pred = model.predict(test_gen)

S1_pred = S_pred[:, 0]
S2_pred = S_pred[:, 1]

# Plot the predictions
plt.plot(T, S1, 'r', label='IVIG')
plt.plot(T, S2, 'g', label='SCIG')
plt.plot(T_test, S1_pred[:len(S1_test)], 'b', label='IVIG Prediction')
plt.plot(T_test, S2_pred[:len(S2_test)], 'y', label='SCIG Prediction')
plt.title('Multivariate LSTM Prediction')
plt.xlabel('Time')
plt.ylabel('Demand')
plt.legend()
plt.show()

mse1 = np.mean((S1[split:] - S1_pred) ** 2)
mse2 = np.mean((S2[split:] - S2_pred) ** 2)

print('IVIG MSE:', mse1)
print('SCIG MSE:', mse2)