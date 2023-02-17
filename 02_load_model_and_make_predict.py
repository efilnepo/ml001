#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 12:03:02 2023

@author: vagrant
"""

# NN engine - tensorflow.keras
import tensorflow as tf
# from tensorflow import keras

#%%

model = tf.keras.models.load_model('./nn1')

# Predict Y for X=10
print(model.predict([10.0]))

