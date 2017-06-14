import numpy as np
from numpy import *

# --------------------------------------------------
# This script is a direct copy of the code provided 
# by Prof. Moon
# --------------------------------------------------

def bit_xor(a,b):
	"""XOR two bits"""
    return int(bool(a)^bool(b))

def error_rate(a,b):
	"""Calculate average hamming loss between two signals"""
    error=np.zeros(len(a))
    for i in range(len(a)):
        error[i]=bit_xor(a[i],b[i])
    return np.sum(error)/len(a)    
    

def bsmc(n,alpha):
	"""Create n-length binary markov chain with switching probability alpha"""
    a=np.zeros(n,dtype=np.int)
    a[0]=int(np.random.random()>0.5)

    for i in range(int(n)-1):
        trans=int(np.random.random()<alpha)
        a[i+1]=bit_xor(a[i],trans) 
    return a

def bsc(x,delta):
	"""Run signal x through a BSC(delta) channel"""
    z=np.zeros(len(x),dtype=np.int)
    for i in range(len(x)):
        noise=int(np.random.random()<delta)
        z[i]=bit_xor(x[i],noise)
    return z
        
def dude(z,k,delta):
	"""Run noisy signal z from BSC(delta) through DUDE algorithm
	
	Arguments
	z -- noisy signal
	k -- context size
	delta -- BSC noise parameter
	
	Output
	x_hat -- estimation based off noisy signal
	"""
    n=len(z)
    x_hat=np.zeros(n,dtype=np.int)

    th_0=2*delta*(1-delta)
    m={}
    
    for i in range(k,n-k):
        context=z[i-k:i].tolist()+z[i+1:i+k+1].tolist()
        context_str = ''.join(str(e) for e in context)
        
        if context_str not in m:
            m[context_str]=np.zeros(2,dtype=np.int)
            m[context_str][z[i]]=1
        else:
            m[context_str][z[i]]+=1
    
    for i in range(k,n-k):
        context=z[i-k:i].tolist()+z[i+1:i+k+1].tolist()
        context_str = ''.join(str(e) for e in context)

        ratio = float(m[context_str][z[i]]) / float(np.sum(m[context_str]))
        if ratio >= th_0:
            x_hat[i]=z[i]
        else:
            x_hat[i]=int(not bool(z[i]))

    return x_hat
 
def dude2(z,k,delta):
	"""Run noisy signal z from BSC(delta) through DUDE algorithm,
	this time using 3-class (0/1/no-change) estimation functions
	
	Arguments:
	z -- noisy signal
	k -- context size
	delta -- BSC noise parameter
	
	Output:
	s_hat -- 3-class estimator indexes based off noisy signal
	m -- contexts
	"""
    n=len(z)
    x_hat=np.zeros(n,dtype=np.int)
    s_hat=x_hat.copy()

    th_0=2*delta*(1-delta)
    th_1=delta**2+(1-delta)**2
    
    m={}
    
    for i in range(k,n-k):
        context=z[i-k:i].tolist()+z[i+1:i+k+1].tolist()
        context_str = ''.join(str(e) for e in context)
        
        if context_str not in m:
            m[context_str]=np.zeros(2,dtype=np.int)
            m[context_str][z[i]]=1
        else:
            m[context_str][z[i]]+=1
    
    for i in range(k,n-k):
        context=z[i-k:i].tolist()+z[i+1:i+k+1].tolist()
        context_str = ''.join(str(e) for e in context)
        
        ratio = float(m[context_str][1]) / float(np.sum(m[context_str]))
        
        if ratio < th_0:
            s_hat[i]=1
        elif ratio >= th_1:
            s_hat[i]=2
        else:
            s_hat[i]=0

    return s_hat, m
     
def denoise_with_s(z,s,k):
	"""Denoise noisy signal z with estimator functions
	
	Arguments:
	z -- noisy signal
	s -- array of estimator indexes
	k - context width
	
	Outputs:
	x_hat -- estimated denoised signal
	"""
    n=len(z)
    x_hat=z.copy()
	# 3 different estimators:
	#    0: don't change
	#    1: set to 0
	#    2: set to 1
	# "change" estimator not necessary
	# as BSC channel has p < 0.5
    for i in range(k,n-k):
        if s[i]==0:
            x_hat[i]=z[i]
        elif s[i]==1:
            x_hat[i]=0
        else:
            x_hat[i]=1
    return x_hat

# --------------------------------------------------
# This function has been replaced by batch_generate
# in main code
# --------------------------------------------------
def make_data_for_ndude(Z,k,L,nb_classes,n):
	"""Generate data for neural net
	
	Arguments:
	Z -- noisy signal
	k -- context width
	L -- loss vector
	nb_classes -- alphabet size (2 for binary)
	n -- signal length
	
	Outputs:
	C -- arranged data matrix for input to nn
	Y -- modified loss vector for use in nn cost function
	"""
    c_length=2*k
    C=np.zeros((n-2*k, 2*k*nb_classes))

    for i in range(k,n-k):
        c_i = vstack((Z[i-k:i,],Z[i+1:i+k+1,])).reshape(1,2*k*nb_classes)
        C[i-k,]=c_i
        
    Y=dot(Z[k:n-k,],L)    
    return C,Y

def make_binary_image(im):
	"""Convert grayscale image to binary image
	
	Arguments:
	im -- 8-bit grayscale image
	
	Outputs:
	im_bin -- binary image
	"""
    im_bin=im.copy()
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i,j]>127: 
                im_bin[i,j]=1
            else:
                im_bin[i,j]=0
    
    return im_bin
