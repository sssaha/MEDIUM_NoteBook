# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.3
#   kernelspec:
#     display_name: Python 3.7.7 64-bit
#     metadata:
#       interpreter:
#         hash: 0bbe64ed72bae4ad15c1d538433b8650b633d7714a535859a2e1e72ed6794111
#     name: python3
# ---

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random

from tsmoothie.smoother import *
from tsmoothie.utils_func import create_windows

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *

from kerashypetune import KerasGridSearch

# +
### READ AND MANAGE DATA ###

df = pd.read_csv('PV_Elec_Gas2.csv', index_col=0)
df.index = pd.to_datetime(df.index)
df['Power'] = df.cum_power.diff()
df.drop(['cum_power'], axis=1, inplace=True)
df.dropna(inplace=True)
print(df.shape)

df.head()

# +
### PLOT RAW DATA ###

# +
### USE KALMAN FILTER TO SMOOTH ALL DATA (ONLY VISUALIZATION PURPOSE) ###

smoother = KalmanSmoother(component='level_longseason', 
                          component_noise={'level':0.1, 'longseason':0.1}, 
                          n_longseasons=365)
smoother.smooth(df[['Elec_kW','Gas_mxm','Power']].T)

# +
### PLOT RAW vs SMOOTHED DATA ###

# +
### TRAIN TEST SPLIT ###

X_train_val, X_test, y_train_val, y_test = train_test_split(df[['Elec_kW','Gas_mxm','Power']].values, df[['Power']].values, 
                                                            test_size=0.2, shuffle=False)

y_train_val.shape, y_test.shape

# +
### USE KALMAN FILTER TO SMOOTH ONLY THE TARGET ON TRAIN/VAL DATA ###

smoother = KalmanSmoother(component='level_longseason', 
                          component_noise={'level':0.1, 'longseason':0.1}, 
                          n_longseasons=365)
smoother.smooth(y_train_val.T)

# +
### SLICE TRAIN/VAL DATA INTO EQUAL SLIDING WINDOWS ###

window_shape = 20
target_seq = 5

X_train_val = create_windows(X_train_val, window_shape=window_shape, 
                             end_id=-target_seq)
X_train_val.shape

# +
### SLICE TEST DATA INTO EQUAL SLIDING WINDOWS ###

X_test = create_windows(X_test, window_shape=window_shape, 
                        end_id=-target_seq)
X_test.shape

# +
### CREATE SLIDING WINDOWS RAW TRAIN/VAL TARGET ###

y_train_val = create_windows(smoother.data.T, window_shape=target_seq, 
                             start_id=window_shape)
y_train_val.shape

# +
### CREATE SLIDING WINDOWS SMOOTH TRAIN/VAL TARGET ###

y_smooth_train_val = create_windows(smoother.smooth_data.T, window_shape=target_seq, 
                                    start_id=window_shape)
y_smooth_train_val.shape

# +
### SLICE TEST TARGET INTO EQUAL SLIDING WINDOWS ###

y_test = create_windows(y_test, window_shape=target_seq, 
                        start_id=window_shape)
y_test.shape

# +
### TRAIN VAL SPLIT ###

X_train, X_val, y_train_raw, y_val_raw = train_test_split(X_train_val, y_train_val, test_size=0.1, shuffle=False)
X_train, X_val, y_train_smooth, y_val_smooth = train_test_split(X_train_val, y_smooth_train_val, test_size=0.1, shuffle=False)

# +
### SCALE INPUT ###

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

# +
### SCALE SMOOTH TARGET ###

scaler_target_smooth = StandardScaler()
y_train_smooth = scaler_target_smooth.fit_transform(y_train_smooth.reshape(-1, y_train_smooth.shape[-1])).reshape(y_train_smooth.shape)
y_val_smooth = scaler_target_smooth.transform(y_val_smooth.reshape(-1, y_val_smooth.shape[-1])).reshape(y_val_smooth.shape)

# +
### SCALE RAW TARGET ###

scaler_target = StandardScaler()
y_train_raw = scaler_target.fit_transform(y_train_raw.reshape(-1, y_train_raw.shape[-1])).reshape(y_train_raw.shape)
y_val_raw = scaler_target.transform(y_val_raw.reshape(-1, y_val_raw.shape[-1])).reshape(y_val_raw.shape)


# +
### UTILITY FUNCTIONS FOR HYPERPARAM SEARCH ###

def set_seed_TF2(seed):
    
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_model(param):

    model = Sequential()
    model.add(LSTM(param['unit'], activation=param['act']))
    model.add(RepeatVector(target_seq))
    model.add(LSTM(param['unit'], activation=param['act'], return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    
    opt_choices = {'adam': Adam(),
                   'rms': RMSprop()}
    
    opt = opt_choices[param['opt']]
    opt.lr = param['lr'] 
    
    model.compile(opt, 'mse')
    
    return model


# +
### CREATE GRID FOR HYPERPARAM SEARCH ###

param_grid = {
    'unit': [128,64,32], 
    'lr': [1e-2,1e-3], 
    'act': ['elu','relu'], 
    'opt': ['adam','rms'],
    'epochs': 200,
    'batch_size': 512
}


# +
### FIT + HYPERPARAM SEARCH WITH SMOOTH TARGET ###

es = EarlyStopping(patience=10, verbose=0, min_delta=0.001, monitor='val_loss', mode='auto', restore_best_weights=True)

hypermodel = get_model
seed_settings = 0
kgs = KerasGridSearch(hypermodel, param_grid, monitor='val_loss', greater_is_better=False, tuner_verbose=1)
kgs.set_seed(set_seed_TF2, seed=seed_settings)
kgs.search(X_train, y_train_smooth, validation_data=(X_val, y_val_smooth), callbacks=[es])

# +
### GET BEST STATISTICS ###

kgs.best_score, kgs.best_params

# +
### REVERSE PREDICTIONS ###

pred_smooth = kgs.best_model.predict(X_test)
pred_smooth = scaler_target_smooth.inverse_transform(pred_smooth.reshape(-1, pred_smooth.shape[-1])).reshape(pred_smooth.shape)

# +
### CALCULATE MSE FOR EACH PREDICTION HORIZONS ###

mse_smooth = {}


for i in range(target_seq):
    
    mse = mean_squared_error(y_test[:,i,0], pred_smooth[:,i,0])
    mse_smooth['day + {i}'.format(i=i+1)] = mse 
    # print(len(pred_smooth[:,i,0]))
    print('pred day + {i}: {mse} MSE'.format(i=i+1, mse=mse))

pd.DataFrame(pred_smooth[:,0,0]).to_csv('output_{}.csv'.format(seed_settings))

# +
### PLOT EACH PREDICTION HORIZON ON THE SAME PLOT ###

plt.figure(figsize=(12,6))

plt.plot(y_test[target_seq:,0,0], c='red', alpha=0.6)

for i in range(target_seq):
        
    plt.plot(pred_smooth[(target_seq-i):-(i+1),i,0], 
             c='blue', alpha=1-1/(target_seq+1)*(i+1),
             label='pred day + {i}'.format(i=i+1))
    
plt.title('prediction w/ smoothing'); plt.legend()




