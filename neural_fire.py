"""
Code to train, validate, save, restore, and apply a 
neural network based fire detector. This code will
only generate a fire mask, not the other products
of the WF-ABBA system. The code is designed to work
with GOES-R (GOES-16) data. 
"""

import keras.models as km
import keras.layers as kl
import goes_vector as gv
import numpy as np 
import numpy.random as nr

def miller_model(training) : 
    n_inputs = len(training.vector)
    n_hidden = int(np.ceil( np.sqrt(n_inputs)))
    model = km.Sequential()
    model.add(kl.Dense(n_hidden, input_shape=(n_inputs,), 
                           use_bias=True,
                           activation='tanh'))
    model.add(kl.Dense(1, activation='sigmoid'))

    return model

def conv_model(training) : 
    model = km.Sequential()
    window_shape = training.normal_windows.shape, 
    model.add(kl.Conv2D(1, 3, data_format='channels_first', 
              input_shape=window_shape,
              activation='tanh',
              use_bias=True))
    model.add(kl.MaxPooling2D(pool_size=window_shape[1:],
                              data_format='channels_first',
                              activation='tanh'))
    model.add(kl.Dense(1, activation='sigmoid'))

    return model


class NeuralFire (object) : 
    def __init__(self,model,score=None, 
                      training=None, training_flag=None,
                      val=None,      val_flag=None,
                      history=None) : 
        self.model         = model
        self.score         = score
        self.training      = training
        self.training_flag = training_flag
        self.val           = val
        self.val_flag      = val_flag
        self.history       = history


    @classmethod
    def train(cls, training_fname, model=None, factory=None, val_frac=0.2, epochs=100) : 
        if ( (model is None) and 
             (factory is None) ) :
            raise ValueError('Must specify either the model or the model factory')

        # read data
        data = gv.GOESVector.read_ncfile(training_fname)

        # set up the model object
        if model is None : 
            model = factory(data[0])
        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])


        # copy data into ndarrays
        vec_data = np.array([ v.normal_windows for v in data ] )
        fireflag = np.array([ v.output for v in data ] ) 

        # partition data into training and validation 
        i_val = np.zeros( (len(data), ), dtype=np.bool)
        n_val = int( np.floor(val_frac*len(data)) ) 
        i = nr.randint(0,len(data), size=(n_val) )
        i_val[i] = True
        training_vec_data = vec_data[np.logical_not(i_val),...]
        training_fireflag = fireflag[np.logical_not(i_val)]
        val_vec_data      = vec_data[i_val,...]
        val_fireflag      = fireflag[i_val]

        # train
        print(training_vec_data.shape)
        print(training_fireflag.shape)
        return
        history=model.fit(training_vec_data, training_fireflag, epochs=epochs, batch_size=32)

        # validate
        score = model.evaluate(val_vec_data, val_fireflag, batch_size=32)

        newobj = cls(model,score,training_vec_data, training_fireflag, 
                     val_vec_data, val_fireflag, history)
        return newobj
