from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.layers.merge import Concatenate
from keras.layers import Embedding
from keras.utils import to_categorical
import keras
import numpy as np

# --------------------------------------------------
# This is our main network creation function
# --------------------------------------------------
def RNNCandACVariable(nb_classes,kmax):
	"""Create the variable-k RNN with left (causal) and right (anti-causal) contexts separate
	
	Arguments
	nb_classes -- The alphabet size
	kmax -- The maximum input length
	"""

    # Create independent left and right context inputs to the model
    inleft = keras.layers.Input(shape=(kmax,))
    inright = keras.layers.Input(shape=(kmax,))
	
    # For each side, create a simply RNN starting model
    Left = Sequential()	# Start with sequential model
    Left.add(Embedding(nb_classes,nb_classes))	# Reshape input into one-hot vector
    Left.add(SimpleRNN(16,activation='tanh'))	# Add RNN with tanh activation
    # Repeat for right side
    Right = Sequential()
    Right.add(Embedding(nb_classes,nb_classes))
    Right.add(SimpleRNN(16,activation='tanh'))
    
	# Join left and right inputs into one layer
    c_state = Left(inleft)	# Grab left and right input networks
    ac_state = Right(inright)
    state = Concatenate(axis=1)([c_state,ac_state])	# Concatenate two sides together
    final_transform = Dense(32,kernel_initializer='he_normal')(state)	# Add a dense node layer
    final_activation = Activation('relu')(final_transform)	# Apply ReLU activation for the layer
    scores = Dense(3,kernel_initializer='he_normal')(final_activation)	# This layer only has 3 nodes, as it is output
    out = Activation('softmax')(scores)	# Output has softmax activation
    model = Model(inputs=[inleft,inright],outputs=out)	# Add all this to the model
    return model

# --------------------------------------------------
# This function, and all the other nn selections, 
# are ultimately obsolete as we use the 
# RNNCandACVariable method for model creation
# and DNN40 for the Neural DUDE evaluation
# --------------------------------------------------
def nn_model_select(C, k, nb_classes, architecture):
	"""Select the model for the neural network.
	
	Arguments:
	C -- The signal which will be denoised, where each row is a context
	k -- Context width
	nb_classes -- Alphabet size (2 for binary)
	architecture -- A string rnn/dnn depicting the type of architecture to use
	"""
    if architecture=="rnn":				# Simple RNN selection
        return C, RNN(k, nb_classes)
    elif architecture=="dnn":			# Simple DNN selection
        return C, DNN40(k,nb_classes)
    else:								# Modified RNN broken into left/right contexts
        return [C[:,0:k*nb_classes],C[:,k*nb_classes:]], RNNCandAC(k,nb_classes)

def DNN40(k, nb_classes):
	"""Create 4-layer 40-hidden-node DNN (copied from Moon's code)
	
	Arguments:
	k -- context width
	nb_classes -- Alphabet size (2 for binary)
	"""
    model=Sequential()	# Start with sequential model
    model.add(Dense(40,input_dim=2*k*nb_classes,kernel_initializer='he_normal'))	# Add a dense node layer
    model.add(Activation('relu'))	# Apply ReLU activation for the layer
    model.add(Dense(40,kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(Dense(40,kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(Dense(3,kernel_initializer='he_normal'))	# This layer only has 3 nodes, as it is output
    model.add(Activation('softmax'))	# Output has softmax activation
    return model

def RNN(k, nb_classes):
	"""Create 3-layer 32-hidden-node DNN with RNN input layer
	
	Arguments:
	k -- context width
	nb_classes -- Alphabet size (2 for binary)
	"""
    model = Sequential()	# Start with sequential model
    model.add(Reshape((2*k,nb_classes),input_shape=(2*k*nb_classes,)))	# Reshape input to match RNN standards
    model.add(SimpleRNN(32, activation='tanh'))	# Create RNN layer with tanh activation
    model.add(Dense(32,kernel_initializer='he_normal'))	# Add a dense node layer
    model.add(Activation('relu'))	# Apply ReLU activation for the layer
    model.add(Dense(3,kernel_initializer='he_normal'))	# This layer only has 3 nodes, as it is output
    model.add(Activation('softmax'))	# Output has softmax activation
    return model

def RNNCandAC(k,nb_classes):
	"""Create RNN model with left (causal) and right (anti-causal) contexts treated differently
	
	Arguments:
	k -- context width
	nb_classes -- Alphabet size (2 for binary)
	"""
    inleft = keras.layers.Input(shape=(k*nb_classes,))	# Create independent left and right context inputs to the model
    inright = keras.layers.Input(shape=(k*nb_classes,))
    l_reshaped = Reshape((k,nb_classes))(inleft)	# Reshape inputs to match RNN form
    r_reshaped = Reshape((k,nb_classes))(inright)
    c_state = SimpleRNN(16,activation='tanh')(l_reshaped)	# Create RNN layers for each side
    ac_state = SimpleRNN(16,activation='tanh',go_backwards=True)(r_reshaped)
    state = Concatenate(axis=1)([c_state,ac_state])	# Concatenate the two RNN outputs
    final_transform = Dense(32,kernel_initializer='he_normal')(state)	# Add a dense layer
    final_activation = Activation('relu')(final_transform)	# Apply ReLU activation for the layer
    scores = Dense(3,kernel_initializer='he_normal')(final_activation)	# This layer only has 3 nodes, as it is output
    out = Activation('softmax')(scores)	# Output has softmax activation
    model = Model(inputs=[inleft,inright],outputs=out)	# Add all this to the model
    return model
    
    