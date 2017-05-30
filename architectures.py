from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.recurrent import SimpleRNN

def DNN40(k, nb_classes):
    model=Sequential()
    model.add(Dense(40,input_dim=2*k*nb_classes,init='he_normal'))
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
    