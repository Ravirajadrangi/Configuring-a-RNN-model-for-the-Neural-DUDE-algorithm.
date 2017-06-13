from keras.preprocessing import sequence
from keras.models import Sequential, Model
#from keras.layers import *
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.layers.merge import Concatenate
from keras.layers import Embedding
from keras.utils import to_categorical
import keras
import numpy as np

def nn_model_select(C, k, nb_classes, architecture):
    if architecture=="rnn":
        return C, RNN(k, nb_classes)
    elif architecture=="dnn":
        return C, DNN40(k,nb_classes)
    else:
        return [C[:,0:k*nb_classes],C[:,k*nb_classes:]], RNNCandAC(k,nb_classes)

def DNN40(k, nb_classes):
    model=Sequential()
    model.add(Dense(40,input_dim=2*k*nb_classes,kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(Dense(40,kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(Dense(40,kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(Dense(3,kernel_initializer='he_normal'))
    model.add(Activation('softmax'))
    return model

def RNN(k, nb_classes):
    #TODO: Create RNN architecture
    model = Sequential()
    model.add(Reshape((2*k,nb_classes),input_shape=(2*k*nb_classes,)))
    model.add(SimpleRNN(32, activation='tanh'))
    model.add(Dense(32,kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(Dense(3,kernel_initializer='he_normal'))
    model.add(Activation('softmax'))
    return model

def RNNCandAC(k,nb_classes):
    inleft = keras.layers.Input(shape=(k*nb_classes,))
    inright = keras.layers.Input(shape=(k*nb_classes,))
    l_reshaped = Reshape((k,nb_classes))(inleft)
    r_reshaped = Reshape((k,nb_classes))(inright)
    c_state = SimpleRNN(16,activation='tanh')(l_reshaped)
    ac_state = SimpleRNN(16,activation='tanh',go_backwards=True)(r_reshaped)
    state = Concatenate(axis=1)([c_state,ac_state])
    final_transform = Dense(32,kernel_initializer='he_normal')(state)
    final_activation = Activation('relu')(final_transform)
    scores = Dense(3,kernel_initializer='he_normal')(final_activation)
    out = Activation('softmax')(scores)
    model = Model(inputs=[inleft,inright],outputs=out)
    return model
        
def RNNCandACVariable(nb_classes,kmax):
    
    inleft = keras.layers.Input(shape=(kmax,))
    inright = keras.layers.Input(shape=(kmax,))
    
    Left = Sequential()
    Left.add(Embedding(nb_classes,nb_classes))
    Left.add(SimpleRNN(16,activation='tanh'))
    
    Right = Sequential()
    Right.add(Embedding(nb_classes,nb_classes))
    Right.add(SimpleRNN(16,activation='tanh'))
    
    c_state = Left(inleft)
    ac_state = Right(inright)
    state = Concatenate(axis=1)([c_state,ac_state])
    final_transform = Dense(32,kernel_initializer='he_normal')(state)
    final_activation = Activation('relu')(final_transform)
    scores = Dense(3,kernel_initializer='he_normal')(final_activation)
    out = Activation('softmax')(scores)
    model = Model(inputs=[inleft,inright],outputs=out)
    return model
    
    