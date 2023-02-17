#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 11:19:05 2023

@author: vagrant
"""

#%% NN engine - tensorflow.keras
# Data representation - numpy
import tensorflow as tf
import numpy as np
from tensorflow import keras

#%%

# Neural network model definition with one layer and one neuron
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# Compile (make actual model engine), loss function - mean_squared_error, optimizer - стохастический градиентный спуск
model.compile(optimizer='sgd', loss='mean_squared_error')

# Make numpy arrays for function Y = 3 X +1
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

#%% Teach the model with 500 retries (epochs)
model.fit(xs, ys, epochs=50)

#%% Predict Y for X=10
print(model.predict([10.0]))

#%% Save
model.save('./nn1')