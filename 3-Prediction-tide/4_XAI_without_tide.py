# === Standard Library ===
import os
import sys
import time
import json
import random
import re
import csv
from datetime import datetime, timedelta
from collections import defaultdict
from glob import glob
from io import StringIO

# === Scientific & Data Libraries ===
import numpy as np
import pandas as pd
import h5py
from scipy.stats import gaussian_kde, linregress
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score

# === Visualization ===
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox
from matplotlib.font_manager import FontProperties
import plotly.graph_objects as go
from PIL import ImageFont

# === TensorFlow / Keras ===
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client import device_lib
from tensorflow.keras import layers, regularizers, backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Dense, BatchNormalization, LSTM, SimpleRNN, Flatten, TimeDistributed,
    Dropout, Activation, Conv1D, Conv2D, Conv3D, Reshape, Input, Concatenate,
    MaxPooling1D, MaxPooling2D, MaxPool3D, RepeatVector, Bidirectional,
    MultiHeadAttention, LayerNormalization
)

# === Networking ===
import requests




for prediction_duration in [12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180]:
#for prediction_duration in [12, 24, 36, 48]:
  print("Loading wind and water level files........")
  wind = h5py.File('wind.h5')
  wl = h5py.File('water_level.h5')

  wind['t'].shape
  wl['t'].shape


  if wl['t'].shape != wind['t'].shape:
      print('WL and wind are not compatible')
      exit()

  all_times = wl['t'][:].astype('datetime64[ms]')


  prediction_duration = prediction_duration
  used_time_behind = prediction_duration

  point = 'SP'
  build_train_set = 1
  ####### Build the train set #################
  if build_train_set == 1:
      x1=[]
      y=[]
      tx=[]

      t=all_times[0]

      print("creating samples....")
      start_indice = 0
      while True:
          indices = np.arange(start_indice,start_indice+used_time_behind,1) #np.where(np.logical_and(t1 <= all_times, all_times < t2))[0]
          w_indices = np.arange(start_indice+used_time_behind,start_indice+prediction_duration+used_time_behind,1) #np.where(np.logical_and(t1 <= all_times, all_times < t2))[0]

          if np.max(indices) >= wind['t'].shape[0]:
              break
          if np.max(w_indices) >= wind['t'].shape[0]:
              break
          x1.append(np.stack((wind['u'][indices], wind['v'][indices], wind['p'][indices]), axis=-1))
          tx.append(all_times[indices])
          y_1=[]

          y_1.append (wl[point]['v'][w_indices])
          y.append(np.stack(y_1,axis=-1))
          start_indice += used_time_behind

      tx = np.stack(tx,axis=0)
      x1 = np.stack(x1,axis=0)
      y  = np.stack(y,axis=0)
      print(str(x1.shape[0]),' Samples created finally')
      print(x1.shape)
      print(y.shape)
      print(tx.shape)

      import numpy as np
      from sklearn.preprocessing import MinMaxScaler

      # --- x1: (4747, 24, 15, 15, 3)
      scaler_x1 = MinMaxScaler()
      x1_reshaped = x1.reshape(-1, x1.shape[-1])   # flatten to (N, 3)
      x1 = scaler_x1.fit_transform(x1_reshaped).reshape(x1.shape)


      # --- y: (4747, 48, 1)
      scaler_y = MinMaxScaler()
      y_reshaped = y.reshape(-1, 1)
      y = scaler_y.fit_transform(y_reshaped).reshape(y.shape)


      print(str(x1.shape[0]),' Samples created finally')
      print(x1.shape, type(x1))
      print(y.shape, type(y))
      print(tx.shape, type(tx))

      np.save('x1.npy',x1)
      np.save('y.npy',y)
      np.save('tx.npy',tx)



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


  #x1 is the sample data for the wind and pressure
  #x1 training data
  #y is the lable

  tx = np.load('tx.npy',allow_pickle=True)

  t_start = np.datetime64('2000-01-01T01:00:00','ms')
  t_end   = np.datetime64('2030-01-01T01:00:00','ms')
  t_filter = (tx >= t_start) & (tx <= t_end)
  t_filter = np.all(t_filter,axis=1)

  x1 = np.load('x1.npy')[t_filter,:]
  y = np.load('y.npy')[t_filter,:]

  ### x1 is the wind and pressure charachteristics
  ### y is the waterlevel prediction (next 24 hours)

  print('x1 shape:', x1.shape)
  print('y shape:', y.shape)

  ## duration ==> n_timesteps
  n_timesteps = y.shape[1]
  ## number of stations ==> n_points
  n_points = y.shape[2]
  ## number of lats or lans ==> n_x
  n_x = x1.shape[2]
  ## number of lats or lans ==> n_y
  n_y = x1.shape[3]
  ## wind v and u directions, pressure ==> n_parameters
  n_parameters = x1.shape[4]
  ## n_samples ==> n_samples
  n_samples = y.shape[0]
  indices = np.arange(n_samples)


  i=0
  train_inputs=[]
  val_inputs=[]
  test_inputs=[]
  train_outputs=[]
  val_outputs=[]
  test_outputs=[]
  test_val_ind = indices[tx[:,0] >= np.datetime64('2020-01-01T00:00:00','ms')]
  val_ind = test_val_ind[:len(test_val_ind)//2]
  test_ind = test_val_ind[len(test_val_ind)//2:]
  train_ind = np.array([j for j in indices if j not in test_val_ind])

  x_train = x1 [train_ind,:,:,:,:]
  x_test  = x1 [test_ind,:,:,:,:]
  x_val   = x1 [val_ind,:,:,:,:]


  y_train =np.array(y[train_ind,:])
  y_test =np.array(y[test_ind,:])
  y_val =np.array(y[val_ind,:])


  n_holdout = 5
  seed = 42
  np.random.seed(seed)
  tf.random.set_seed(seed)
  epochs = 50


  trainr_1= []
  valr_1 = []
  trainmae_1= []
  valmae_1 = []
  trainrmse_1= []
  valrmse_1 = []
  trainmse_1= []
  valmse_1 = []
  testl_1= []
  testr_1 = []
  testmae_1 = []
  testrmse_1= []
  testmse_1 = []
  t1 = []

  for j in range(n_holdout):
      input_wind = Input(shape=(n_timesteps, n_x, n_y, n_parameters))

      # Increase number of filters in Conv2D layers
      x = TimeDistributed(Conv2D(64, (3, 3), padding='same'))(input_wind)
      x = BatchNormalization()(x)
      x = Activation('relu')(x)
      x = TimeDistributed(Conv2D(32, (3, 3), padding='same'))(x)
      x = BatchNormalization()(x)
      x = Activation('relu')(x)

      # Adjust the Dense layers for more complexity
      x = Reshape((n_timesteps, -1))(x)
      x = Dense(128, activation='relu')(x)
      x = BatchNormalization()(x)
      x = Dense(32, activation='relu')(x)

      # Increase the dropout rate for better regularization
      x = Dropout(0.2)(x)

      # Increase the number of units in LSTM layers
      x = LSTM(256, return_sequences=True)(x)

      x = Dense(32, activation='relu')(x)
      x = Dropout(0.2)(x)
      x = Dense(1, activation='linear')(x)

      model = Model(inputs=[input_wind], outputs=x)
      model.summary()
      model.compile(
          optimizer=tf.keras.optimizers.Adam(),
          loss=mse_metric,
          metrics=[r_squared, mae_metric, rmse_metric, mse_metric]
      )

      checkpoint = ModelCheckpoint(f'best_model_iter_{j}.h5',
                                  monitor='val_mse_metric',
                                  save_best_only=True,
                                  mode='min',
                                  verbose=1)

      start_time = time.time()
      hist = model.fit([x_train], y_train,
                      batch_size=64, epochs=epochs, verbose=1,
                      validation_data=([x_val], y_val),
                      callbacks=[checkpoint])
      t1.append(time.time() - start_time)

      # Evaluate: returns [loss, r_squared, mae, rmse, mse]
      test_vals = model.evaluate([x_test], y_test, batch_size=64, verbose=1)

      model.save(f'model1{j}.h5')
      # Save curves
      trainr_1.append(hist.history['r_squared'])
      valr_1.append(hist.history['val_r_squared'])
      trainmae_1.append(hist.history['mae_metric'])
      valmae_1.append(hist.history['val_mae_metric'])
      trainrmse_1.append(hist.history['rmse_metric'])
      valrmse_1.append(hist.history['val_rmse_metric'])
      trainmse_1.append(hist.history['mse_metric'])
      valmse_1.append(hist.history['val_mse_metric'])

      # Save test
      testr_1.append(test_vals[1])
      testmae_1.append(test_vals[2])
      testrmse_1.append(test_vals[3])
      testmse_1.append(test_vals[4])


  # to numpy
  trainr_1 = np.array(trainr_1)
  valr_1 = np.array(valr_1)
  trainmae_1= np.array(trainmae_1)
  valmae_1 = np.array(valmae_1)
  trainrmse_1 = np.array(trainrmse_1)
  valrmse_1 = np.array(valrmse_1)
  trainmse_1 = np.array(trainmse_1)
  valmse_1 =  np.array(valmse_1)

  # pack training curves: (n_holdout, epochs, 10)
  train1 = np.stack(
      (trainr_1, valr_1,
      trainmae_1, valmae_1, trainrmse_1, valrmse_1,
      trainmse_1, valmse_1),
      axis=-1
  )

  # pack test metrics: (n_holdout, 6) -> loss, r2, mae, rmse, mse, time
  testr_1 = np.array(testr_1)
  testmae_1, testrmse_1, testmse_1 = np.array(testmae_1), np.array(testrmse_1), np.array(testmse_1)
  t1 = np.array(t1)
  test1 = np.stack((testr_1, testmae_1, testrmse_1, testmse_1, t1), axis=-1)

  np.save('train1.npy', train1)
  np.save('test1.npy', test1)



  trainr_1= []
  valr_1 = []
  trainmae_1= []
  valmae_1 = []
  trainrmse_1= []
  valrmse_1 = []
  trainmse_1= []
  valmse_1 = []
  testl_1= []
  testr_1 = []
  testmae_1 = []
  testrmse_1= []
  testmse_1 = []
  t1 = []

  for j in range(n_holdout):
      input_wind = Input(shape=(n_timesteps, 15, 15, 3))
      x = Reshape((n_timesteps, n_parameters*n_x*n_y))(input_wind)
      x = LSTM(128, return_sequences=True)(x)
      x = Dense(16, activation='relu')(x)
      x = Dropout(0.1)(x)
      x = LSTM(128, return_sequences=True)(x)
      x = Concatenate()([x])
      x = Dense(16, activation='relu')(x)
      x = Dropout(0.1)(x)
      x = Dense(1, activation='linear')(x)
      model  = Model(inputs=[input_wind], outputs=x)
      model.summary()
      model.compile(
          optimizer=tf.keras.optimizers.Adam(),
          loss=mse_metric,
          metrics=[r_squared, mae_metric, rmse_metric, mse_metric]
      )

      checkpoint = ModelCheckpoint(f'best_model_iter_{j}.h5',
                                  monitor='val_mse_metric',
                                  save_best_only=True,
                                  mode='min',
                                  verbose=1)

      start_time = time.time()
      hist = model.fit([x_train], y_train,
                      batch_size=64, epochs=epochs, verbose=1,
                      validation_data=([x_val], y_val),
                      callbacks=[checkpoint])
      t1.append(time.time() - start_time)

      # Evaluate: returns [loss, r_squared, mae, rmse, mse]
      test_vals = model.evaluate([x_test], y_test, batch_size=64, verbose=1)

      model.save(f'model2{j}.h5')
      # Save curves
      trainr_1.append(hist.history['r_squared'])
      valr_1.append(hist.history['val_r_squared'])
      trainmae_1.append(hist.history['mae_metric'])
      valmae_1.append(hist.history['val_mae_metric'])
      trainrmse_1.append(hist.history['rmse_metric'])
      valrmse_1.append(hist.history['val_rmse_metric'])
      trainmse_1.append(hist.history['mse_metric'])
      valmse_1.append(hist.history['val_mse_metric'])

      # Save test
      testr_1.append(test_vals[1])
      testmae_1.append(test_vals[2])
      testrmse_1.append(test_vals[3])
      testmse_1.append(test_vals[4])

  # to numpy
  trainr_1 = np.array(trainr_1)
  valr_1 = np.array(valr_1)
  trainmae_1= np.array(trainmae_1)
  valmae_1 = np.array(valmae_1)
  trainrmse_1 = np.array(trainrmse_1)
  valrmse_1 = np.array(valrmse_1)
  trainmse_1 = np.array(trainmse_1)
  valmse_1 =  np.array(valmse_1)

  # pack training curves: (n_holdout, epochs, 10)
  train1 = np.stack(
      (trainr_1, valr_1,
      trainmae_1, valmae_1, trainrmse_1, valrmse_1,
      trainmse_1, valmse_1),
      axis=-1
  )

  # pack test metrics: (n_holdout, 6) -> loss, r2, mae, rmse, mse, time
  testr_1 = np.array(testr_1)
  testmae_1, testrmse_1, testmse_1 = np.array(testmae_1), np.array(testrmse_1), np.array(testmse_1)
  t1 = np.array(t1)
  test1 = np.stack((testr_1, testmae_1, testrmse_1, testmse_1, t1), axis=-1)

  np.save('train2.npy', train1)
  np.save('test2.npy', test1)




  trainr_1= []
  valr_1 = []
  trainmae_1= []
  valmae_1 = []
  trainrmse_1= []
  valrmse_1 = []
  trainmse_1= []
  valmse_1 = []
  testl_1= []
  testr_1 = []
  testmae_1 = []
  testrmse_1= []
  testmse_1 = []
  t1 = []
  for j in range(n_holdout):
      input_wind = Input(shape=(n_timesteps, n_x, n_y, n_parameters))
      x = Conv3D(64, (3, 3, 3), padding='same')(input_wind)
      x = BatchNormalization()(x)
      x = Activation('relu')(x)
      x = Conv3D(32, (3, 3, 3), padding='same')(x)
      x = BatchNormalization()(x)
      x = Activation('relu')(x)
      x = Conv3D(16, (3, 3, 3), padding='same')(x)
      x = BatchNormalization()(x)
      x = Activation('relu')(x)
      x = Reshape((n_timesteps, -1))(x)
      x = Dense(64, activation='relu')(x)
      x = Dense(32, activation='relu')(x)
      x = Dropout(0.1)(x)
      x = Dense(16, activation='relu')(x)
      x = Dropout(0.1)(x)
      x = Dense(1, activation='linear')(x)
      x = Reshape((n_timesteps,1))(x)
      model = Model(inputs=[input_wind], outputs=x)
      model.summary()
      model.compile(
          optimizer=tf.keras.optimizers.Adam(),
          loss=mse_metric,
          metrics=[r_squared, mae_metric, rmse_metric, mse_metric]
      )

      checkpoint = ModelCheckpoint(f'best_model_iter_{j}.h5',
                                  monitor='val_mse_metric',
                                  save_best_only=True,
                                  mode='min',
                                  verbose=1)

      start_time = time.time()
      hist = model.fit([x_train], y_train,
                      batch_size=64, epochs=epochs, verbose=1,
                      validation_data=([x_val], y_val),
                      callbacks=[checkpoint])
      t1.append(time.time() - start_time)

      # Evaluate: returns [loss, r_squared, mae, rmse, mse]
      test_vals = model.evaluate([x_test], y_test, batch_size=64, verbose=1)

      model.save(f'model3{j}.h5')
      # Save curves
      trainr_1.append(hist.history['r_squared'])
      valr_1.append(hist.history['val_r_squared'])
      trainmae_1.append(hist.history['mae_metric'])
      valmae_1.append(hist.history['val_mae_metric'])
      trainrmse_1.append(hist.history['rmse_metric'])
      valrmse_1.append(hist.history['val_rmse_metric'])
      trainmse_1.append(hist.history['mse_metric'])
      valmse_1.append(hist.history['val_mse_metric'])

      # Save test
      testr_1.append(test_vals[1])
      testmae_1.append(test_vals[2])
      testrmse_1.append(test_vals[3])
      testmse_1.append(test_vals[4])

  # to numpy
  trainr_1 = np.array(trainr_1)
  valr_1 = np.array(valr_1)
  trainmae_1= np.array(trainmae_1)
  valmae_1 = np.array(valmae_1)
  trainrmse_1 = np.array(trainrmse_1)
  valrmse_1 = np.array(valrmse_1)
  trainmse_1 = np.array(trainmse_1)
  valmse_1 =  np.array(valmse_1)

  # pack training curves: (n_holdout, epochs, 10)
  train1 = np.stack(
      (trainr_1, valr_1,
      trainmae_1, valmae_1, trainrmse_1, valrmse_1,
      trainmse_1, valmse_1),
      axis=-1
  )

  # pack test metrics: (n_holdout, 6) -> loss, r2, mae, rmse, mse, time
  testr_1 = np.array(testr_1)
  testmae_1, testrmse_1, testmse_1 = np.array(testmae_1), np.array(testrmse_1), np.array(testmse_1)
  t1 = np.array(t1)
  test1 = np.stack((testr_1, testmae_1, testrmse_1, testmse_1, t1), axis=-1)

  np.save('train3.npy', train1)
  np.save('test3.npy', test1)



  ###Plots




  ########################################## Figure 3 #########################################################
  train1 = np.load('train1.npy')
  train2 = np.load('train2.npy')
  train3 = np.load('train3.npy')


  test1 = np.load('test1.npy')
  test2 = np.load('test2.npy')
  test3 = np.load('test3.npy')


  ### Table

  ########################################## Table 1 #########################################################
  Header = ['Model', 'Train_MAE', 'Train_RMSE', 'Train_MSE', 'Train_R2','Test_MAE', 'Test_RMSE', 'Test_MSE', 'Test_R2','Training Time']
  cnnlstm = ['CNN-LSTM', np.mean(train1,axis=0)[-1,2], np.mean(train1,axis=0)[-1,4], np.mean(train1,axis=0)[-1,6], np.mean(train1,axis=0)[-1,0],
            np.mean(test1,axis=0)[1], np.mean(test1,axis=0)[2], np.mean(test1,axis=0)[3], np.mean(test1,axis=0)[0], np.mean(test1,axis=0)[4]]
  lstm = ['LSTM', np.mean(train2,axis=0)[-1,2], np.mean(train2,axis=0)[-1,4], np.mean(train2,axis=0)[-1,6], np.mean(train2,axis=0)[-1,0],
            np.mean(test2,axis=0)[1], np.mean(test2,axis=0)[2], np.mean(test2,axis=0)[3], np.mean(test2,axis=0)[0], np.mean(test2,axis=0)[4]]
  cnn = ['3DCNN', np.mean(train3,axis=0)[-1,2], np.mean(train3,axis=0)[-1,4], np.mean(train3,axis=0)[-1,6], np.mean(train3,axis=0)[-1,0],
            np.mean(test3,axis=0)[1], np.mean(test3,axis=0)[2], np.mean(test3,axis=0)[3], np.mean(test3,axis=0)[0], np.mean(test3,axis=0)[4]]
  table = [Header, cnnlstm, lstm, cnn]

  with open('table'+str(prediction_duration)+'.csv', mode='w', newline='') as file:
      writer = csv.writer(file)
      writer.writerows(table)
