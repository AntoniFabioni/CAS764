import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.gaussian_process as gp

data = pd.read_csv('simulated_data.csv')

T = np.array(data['T'])
S1 = np.array(data['IVIG'])
S2 = np.array(data['SCIG'])

# kernel = gp.kernels.ConstantKernel(1.0) * gp.kernels.RBF(length_scale=1.0)
# kernel = gp.kernels.ConstantKernel(1.0) * gp.kernels.RBF(length_scale=1.0) + gp.kernels.WhiteKernel(noise_level=1.0)
kernel = gp.kernels.ConstantKernel(1.0) * gp.kernels.RBF(length_scale=1.0) + gp.kernels.WhiteKernel(noise_level=1.0) + gp.kernels.RationalQuadratic(alpha=0.1, length_scale=1.0)

model1 = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=True)
model2 = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=True)

model1.fit(T[1:300, np.newaxis], S1[1:300])
model2.fit(T[1:300, np.newaxis], S2[1:300])

T_pred = np.linspace(0, len(T), 1000)
S1_pred, S1_pred_std = model1.predict(T_pred[:, np.newaxis], return_std=True)
S2_pred, S2_pred_std = model2.predict(T_pred[:, np.newaxis], return_std=True)

S1_pred_std = 1.96 * S1_pred_std
S2_pred_std = 1.96 * S2_pred_std

plt.plot(T, S1, 'r', label='IVIG')
plt.plot(T_pred, S1_pred, 'b', label='IVIG Prediction')
plt.plot(T, S2, 'g', label='SCIG')
plt.plot(T_pred, S2_pred, 'y', label='SCIG Prediction')
plt.fill_between(T_pred, S1_pred - S1_pred_std, S1_pred + S1_pred_std, alpha=0.2, color='b')
plt.fill_between(T_pred, S2_pred - S2_pred_std, S2_pred + S2_pred_std, alpha=0.2, color='y')
plt.title('Gaussian Process Prediction')
plt.xlabel('Time')
plt.ylabel('Demand')
plt.legend()
plt.show()

S1_pred = np.interp(T, T_pred, S1_pred)
S2_pred = np.interp(T, T_pred, S2_pred)

mse1 = np.mean((S1 - S1_pred) ** 2)
mse2 = np.mean((S2 - S2_pred) ** 2)

print('IVIG MSE:', mse1)
print('SCIG MSE:', mse2)