import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts.models import ExponentialSmoothing
from darts import TimeSeries

data = pd.read_csv('simulated_data.csv')

T = np.array(data['T'])
S1 = np.array(data['IVIG'])
S2 = np.array(data['SCIG'])

series1 = TimeSeries.from_values(S1)
series2 = TimeSeries.from_values(S2)

train1, val1 = series1[:400], series1[400:]
train2, val2 = series2[:400], series2[400:]

model1 = ExponentialSmoothing()
model2 = ExponentialSmoothing()

model1.fit(train1)
model2.fit(train2)

S1_pred = model1.predict(len(val1))
S2_pred = model2.predict(len(val2))

mse1 = np.mean((val1.values() - S1_pred.values()) ** 2)
mse2 = np.mean((val2.values() - S2_pred.values()) ** 2)

plt.plot(T, S1, 'r', label='IVIG')
plt.plot(T[400:], S1_pred.values(), 'b', label='IVIG Prediction')
plt.plot(T, S2, 'g', label='SCIG')
plt.plot(T[400:], S2_pred.values(), 'y', label='SCIG Prediction')
plt.title('Exponential Smoothing Prediction')
plt.xlabel('Time')
plt.ylabel('Demand')
plt.legend()
plt.show()

print('IVIG MSE:', mse1)
print('SCIG MSE:', mse2)