import pdb

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
import numpy as np



class Basic(tf.keras.Model):

    def __init__(self, n_units, n_hidden, dropout_rate):
        super(Basic, self).__init__()
        self.n_units = n_units
        self.n_hidden = n_hidden
        self.dropout_rate = dropout_rate
        self.Dens = list()
        self.BN = list()
        self.Drop = list()
        for i in np.arange(n_hidden):
            if i == 0:
                self.Dens.append(layers.Dense(n_units, activation='relu'))
            else:
                self.Dens.append(layers.Dense(n_units, activation='relu'))
            self.BN.append(layers.BatchNormalization())
            self.Drop.append(layers.Dropout(dropout_rate))
        self.dens_last = layers.Dense(1)
        # self.BN_last = layers.BatchNormalization()
        # self.sigmoid = activations.sigmoid()

    def call(self, inputs):
        for i in np.arange(len(self.Dens)):
            if i == 0:
                x = self.Dens[i](inputs)
            else:
                x = self.Dens[i](x)
            x = self.BN[i](x)
            x = self.Drop[i](x)
        x = self.dens_last(x)
        # x = self.BN_last(x)
        return x

    def copy(self):
        copy = Basic(self.n_units, self.n_hidden, self.dropout_rate)
        input_dim = self.layers[0].weights[0].shape[0]
        copy.build((None, input_dim))
        for l1, l2 in zip(self.layers, copy.layers):
            l2.set_weights(l1.get_weights( ))
        return copy


class BasicSigmoid(tf.keras.Model):

    def __init__(self, n_units, n_hidden, dropout_rate):
        super(BasicSigmoid, self).__init__()
        self.n_units = n_units
        self.n_hidden = n_hidden
        self.dropout_rate = dropout_rate
        self.Dens = list()
        self.BN = list()
        self.Drop = list()
        for i in np.arange(n_hidden):
            if i == 0:
                self.Dens.append(layers.Dense(n_units, activation='relu'))
            else:
                self.Dens.append(layers.Dense(n_units, activation='relu'))
            self.BN.append(layers.BatchNormalization())
            self.Drop.append(layers.Dropout(dropout_rate))
        self.dens_last = layers.Dense(1, activation='sigmoid')
        # self.BN_last = layers.BatchNormalization()
        #self.sigmoid = activations.sigmoid()

    def call(self, inputs):
        for i in np.arange(len(self.Dens)):
            if i == 0:
                x = self.Dens[i](inputs)
            else:
                x = self.Dens[i](x)
            x = self.BN[i](x)
            x = self.Drop[i](x)
        x = self.dens_last(x)
        # x = self.BN_last(x)
        return x

    def new(self):
        copy = BasicSigmoid(self.n_units, self.n_hidden, self.dropout_rate)
        input_dim = self.layers[0].weights[0].shape[0]
        copy.build((None, input_dim))
        return copy

    def copy(self):
        #pdb.set_trace()
        copy = BasicSigmoid(self.n_units, self.n_hidden, self.dropout_rate)
        input_dim = self.layers[0].weights[0].shape[0]
        copy.build((None, input_dim))
        for l1, l2 in zip(self.layers, copy.layers):
            l2.set_weights(l1.get_weights( ))
        return copy


class BasicRelu(tf.keras.Model):

    def __init__(self, n_units, n_hidden, dropout_rate):
        super(BasicRelu, self).__init__()
        self.n_units = n_units
        self.n_hidden = n_hidden
        self.dropout_rate = dropout_rate
        self.Dens = list()
        self.BN = list()
        self.Drop = list()
        for i in np.arange(n_hidden):
            if i == 0:
                self.Dens.append(layers.Dense(n_units, activation='relu'))
            else:
                self.Dens.append(layers.Dense(n_units, activation='relu'))
            self.BN.append(layers.BatchNormalization())
            self.Drop.append(layers.Dropout(dropout_rate))
        self.dens_last = layers.Dense(1)
        # self.BN_last = layers.BatchNormalization()
        # self.sigmoid = activations.sigmoid()

    def call(self, inputs):
        for i in np.arange(len(self.Dens)):
            if i == 0:
                x = self.Dens[i](inputs)
            else:
                x = self.Dens[i](x)
            x = self.BN[i](x)
            x = self.Drop[i](x)
        x = self.dens_last(x)
        # x = self.BN_last(x)
        return activations.relu(x)

    def copy(self):
        copy = BasicRelu(self.n_units, self.n_hidden, self.dropout_rate)
        input_dim = self.layers[0].weights[0].shape[0]
        copy.build((None, input_dim))
        for l1, l2 in zip(self.layers, copy.layers):
            l2.set_weights(l1.get_weights( ))
        return copy

