
########################################## Figure 4 #########################################################
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from scipy.stats import pearsonr


def cc(y_true, y_pred):
    corr_coefficient, _ = pearsonr(y_true, y_pred)
    return ( corr_coefficient )

def r_squared(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1.0 - ss_res / (ss_tot + tf.keras.backend.epsilon())

def mae_metric(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def rmse_metric(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

def mse_metric(y_true, y_pred):
    mse_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    return mse_loss


datum = 'MSL'
station='8726520'

start_time=np.datetime64('2022-09-27T12:00:00')
end_time  =np.datetime64('2022-09-30T00:00:00')


wind = h5py.File('wind.h5')
wl = h5py.File('water_level.h5')

wind['t'].shape
wl['t'].shape


if wl['t'].shape != wind['t'].shape:
    print('WL and wind are not compatible')
    exit()

all_times = wl['t'][:].astype('datetime64[ms]')

prediction_duration = 24
used_time_behind = prediction_duration

point = 'SP'
build_train_set = 1
####### Build the train set #################

# global x1,x2,y,tx
x1=[]
x2=[]
y=[]
tx=[]

t=all_times[0]
start_indice = np.where(all_times == start_time)[0][0]

print("creating samples....")
while True:
    indices = np.arange(start_indice,start_indice+used_time_behind,1) 
    w_indices = np.arange(start_indice+used_time_behind,start_indice+prediction_duration+used_time_behind,1) 

    if np.max(indices) >= wind['t'].shape[0]:
        break
    if np.max(w_indices) >= wind['t'].shape[0]:
        break

    x1.append(np.stack((wind['u'][indices], wind['v'][indices], wind['p'][indices]), axis=-1))
    tx.append(all_times[indices])
    y_1=[]
    x2_1=[]
    x2_1.append(wl[point]['p'][indices])
    y_1.append (wl[point]['v'][w_indices])

    x2.append(np.stack(x2_1,axis=-1))
    y.append(np.stack(y_1,axis=-1))
    if all_times[indices][-1] > end_time:
        break
    else:
        start_indice += used_time_behind



tx = np.stack(tx,axis=0)
x1 = np.stack(x1,axis=0)
x2 = np.stack(x2,axis=0)
y  = np.stack(y,axis=0)

print('x1',x1.shape)
print(y.shape)



# --- x1: (4747, 24, 15, 15, 3)
scaler_x1 = MinMaxScaler()
x1_reshaped = x1.reshape(-1, x1.shape[-1])   # flatten to (N, 3)
x1 = scaler_x1.fit_transform(x1_reshaped).reshape(x1.shape)


# --- x2: (4747, 48, 1)
scaler_x2 = MinMaxScaler()
x2_reshaped = x2.reshape(-1, 1)
x2 = scaler_x2.fit_transform(x2_reshaped).reshape(x2.shape)


# --- y: (4747, 48, 1)
scaler_y = MinMaxScaler()
y_reshaped = y.reshape(-1, 1)
y = scaler_y.fit_transform(y_reshaped).reshape(y.shape)




y_pred1 = []
for j in range (5):
    model = tf.keras.models.load_model(f'./model1{j}.h5', custom_objects={"mse_metric": mse_metric, "r_squared": r_squared})
    y_pred1.append(model.predict([x1, x2],verbose=0))
y_pred1 = np.mean(np.array(y_pred1),axis = 0)

y_pred2 = []
for j in range (5):
    model = tf.keras.models.load_model(f'./model2{j}.h5', custom_objects={"mse_metric": mse_metric, "r_squared": r_squared})
    y_pred2.append(model.predict([x1, x2],verbose=0))
y_pred2 = np.mean(np.array(y_pred2),axis = 0)

y_pred3 = []
for j in range (5):
    model = tf.keras.models.load_model(f'./model3{j}.h5', custom_objects={"mse_metric": mse_metric, "r_squared": r_squared})
    y_pred3.append(model.predict([x1, x2],verbose=0))
y_pred3 = np.mean(np.array(y_pred3),axis = 0)

print('y_pred1', y_pred1.shape)

y_pred1 = y_pred1[:, :, 0].flatten()
y_pred2 = y_pred2[:, :, 0].flatten()
y_pred3 = y_pred3[:, :, 0].flatten()
t_plot = tx.flatten()
y2 = y.flatten()
print('y_pred1', y_pred1.shape)
print('y2', y2.shape)


import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# === Match the Seaborn style from your "Test MSE vs Window Size" plot ===
sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
sns.set_palette("deep")  # same as the Excel/MSE example

# === Plot ===
plt.figure(figsize=(8, 5))

sns.lineplot(x=t_plot, y=y2, label='Verified Water Level', marker='o', linewidth=1.5)
sns.lineplot(x=t_plot, y=y_pred1, label='CNN-LSTM', marker='s', linewidth=1.5)
sns.lineplot(x=t_plot, y=y_pred2, label='LSTM', marker='^', linewidth=1.5)
sns.lineplot(x=t_plot, y=y_pred3, label='3D-CNN', marker='D', linewidth=1.5)

# === Formatting (same as MSE plot) ===
plt.xlim(t_plot[0], t_plot[-1])
plt.title("Water Level Prediction During Hurricane by Different Methods", fontsize=14, fontweight='bold')
plt.xlabel("Time", fontsize=14)
plt.ylabel("Water Level (m)", fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.grid(True)
plt.legend(title="Model", fontsize=10, title_fontsize=11, loc='lower right')
plt.tight_layout()

# === Save and show ===
plt.savefig("Surge_DifferentMethods.pdf", dpi=300, bbox_inches='tight')
plt.show()



########################################## Table 1 #########################################################
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

Header = ['Model', 'Surge_MAE', 'Surge_RMSE', 'Surge_MSE', 'Surge_R2']
cnnlstm = ['CNN-LSTM', mean_absolute_error(y2, y_pred1), np.sqrt(mean_squared_error(y2, y_pred1)), mean_squared_error(y2, y_pred1), cc(y2, y_pred1)]
lstm = ['LSTM', mean_absolute_error(y2, y_pred2), np.sqrt(mean_squared_error(y2, y_pred2)), mean_squared_error(y2, y_pred2), cc(y2, y_pred2)]
cnn = ['3DCNN', mean_absolute_error(y2, y_pred3), np.sqrt(mean_squared_error(y2, y_pred3)), mean_squared_error(y2, y_pred3), cc(y2, y_pred3)]

table = [Header,cnnlstm,lstm,cnn]

with open('table_surge.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(table)

import pandas as pd

# Read the CSV file
df = pd.read_csv("table_surge.csv")

# Round all numeric values to 4 decimals
df = df.round(4)

# Convert to LaTeX table
latex_table = df.to_latex(index=False, escape=False)

# Save to file
with open("table_surge.tex", "w", encoding="utf-8") as f:
    f.write(latex_table)

print("✅ LaTeX table exported to 'tablesurge.tex'")
