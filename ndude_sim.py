import os
import numpy as np
import binary_dude as bd

from numpy import *

import keras

from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.optimizers import SGD, RMSprop, Adagrad, Adam, Adadelta
from keras.utils import np_utils

from architectures import *

from PIL import Image


def simulate_ndude(source="Einstein256_256", im_dat=True, alpha=0.1, delta=0.1, k_max=40, nn_dat=True, foldername=None, relmodel=False, savemodel=True):
	nb_classes=2
	L=np.array([[delta, -delta/(1-2*delta), (1-delta)/(1-2*delta)],[delta, (1-delta)/(1-2*delta), -delta/(1-2*delta)]])
	L_new=-L+(1-delta)/(1-2*delta)

	print('Lnew:', L_new)

# -----------------------------------------------------
# Generate/Load clean/noisy image data
# -----------------------------------------------------

	if im_dat == True:
		if os.path.isfile('imdat/' + source + '_data.npz') == True:
			npzfile = np.load('imdat/' + source + '_data.npz')
			x=npzfile['x']
			z=npzfile['z']
		else:
			print('WARNING: NP data requested but not found')
			im_dat = False

	if im_dat == False:
		fexts = ['.jpg', '.png']
		ffound = False
		fext = '.jpg'
		for ext in fexts:
			if os.path.isfile(source + ext):
				ffound = True
				fext = ext
				break
		if ffound == True:
			im=Image.open(source+fext).convert('L')
			imarray=np.array(im)
			n=imarray.shape[0]*imarray.shape[1]
			im_bin=bd.make_binary_image(imarray)
			x=im_bin.copy().reshape(n,)
			z=bd.bsc(x,delta)
			np.savez('imdat/' + source + '_data', x=x, z=z)
		im_dat = ffound

	if im_dat == False:
		print('ERROR: Cannot find test data')
		exit()

	n=x.shape[0]

	Z=np_utils.to_categorical(z,nb_classes)
	err_nn_k=zeros(k_max)
	err_dude_k=zeros(k_max)

	err_dude_k[0]=delta
	err_nn_k[0]=delta

	est_loss=zeros((2,k_max))
	est_loss[0,0]=0.1
	est_loss[1,0]=0.1

	x_hat_dude=np.zeros((k_max,n))
	x_hat_n_dude=np.zeros((k_max,n))

	k_range=range(1,k_max+1)
	print('k range: ', k_range)

	for k in k_range:
		print('k=',k)

# -----------------------------------------------------
# For getting data for Neural DUDE
# -----------------------------------------------------
		if nn_dat == True and os.path.isfile('nndat/'+source+'_nn_dat_'+str(k)+'.npz') == True:
			file_n = 'nndat/'+source+'_nn_dat_'+str(k)+'.npz'
			nn_data = np.load(file_n)
			C = nn_data['C']
			Y = nn_data['Y']
		else:
			C,Y = bd.make_data_for_ndude(Z,k,L_new,nb_classes,n)
			np.savez('nndat/'+source+'_nn_dat_'+str(k), C=C, Y=Y)

# -----------------------------------------------------
# Defining neural networks
# -----------------------------------------------------
		if relmodel == True:
			if os.path.isfile('models/'+source+'_k_'+str(k)+'.h5') == True:
				model = load_model('models/'+source+'_k_'+str(k)+'.h5')
			else:
				print('WARNING: Model data requested but not found')
				relmodel = False

		if relmodel == False:
			model = nn_model_select(k, nb_classes)

			rms=RMSprop(lr=0.001, rho=0.9, epsilon=1e-06,clipnorm=1.5)
			adagrad=Adagrad(clipnorm=1.5)
			adam=Adam()
			adadelta=Adadelta()
			sgd=SGD(lr=0.01,decay=1e-6,momentum=0.95, nesterov=True, clipnorm=1.0)

			model.compile(loss='poisson', optimizer=adam)

			print('Model fitting...')
			model.fit(C,Y,epochs=10,batch_size=100, verbose=0,
				validation_data=(C, Y))
			if savemodel == True:
				print('Saving model...')
				model.save('models/'+source+'_k_'+str(k)+'.h5')

#-------------------------------------------------------------------------------------
		pred_class=model.predict_classes(C, batch_size=200, verbose=0)
		s_nn_hat=hstack((zeros(k),pred_class,zeros(k)))
		x_nn_hat=bd.denoise_with_s(z,s_nn_hat,k)
		error_nn=bd.error_rate(x,x_nn_hat)
		print('error_nn=', error_nn)
		err_nn_k[k-1]=error_nn

		x_hat_n_dude[k-1,:]=x_nn_hat


		s_hat,m= bd.dude2(z,k,delta) 
		x_dude_hat=bd.denoise_with_s(z,s_hat,k)
		error_dude=bd.error_rate(x,x_dude_hat)
		print('error_dude=',error_dude)
		err_dude_k[k-1]=error_dude

		x_hat_dude[k-1,:]=x_dude_hat

		s_class=3
		s_hat_cat=np_utils.to_categorical(s_hat,s_class)
		s_nn_hat_cat=np_utils.to_categorical(s_nn_hat,s_class)

		emp_dist=dot(Z,L)
		est_loss_dude=mean(sum(emp_dist*s_hat_cat,axis=1))
		est_loss_nn_dude=mean(sum(emp_dist*s_nn_hat_cat,axis=1))

		est_loss[0,k-1]=est_loss_dude
		est_loss[1,k-1]=est_loss_nn_dude

#-------------------------------------------------------------------------------------
	print('Finished error evaluation')
	return k_range, est_loss, err_dude_k, err_nn_k, x_dude_hat, x_nn_hat, x, z

def save_ndude_data(source, k, est_loss, err_dude_k, err_nn_k):
	index = 1
	while os.path.isfile('ndude_configs/NDUDE_Data_'+source+'_'+str(index)+'.npz') == True:
		index += 1

	np.savez('ndude_configs/NDUDE_Data_'+source+'_'+str(index), k=k, est_loss=est_loss, err_dude_k=err_dude_k, err_nn_k=err_nn_k)