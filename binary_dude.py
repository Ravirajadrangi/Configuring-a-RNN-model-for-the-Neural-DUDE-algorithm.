
# coding: utf-8

# In[1]:

import numpy as np
from numpy import *

# In[3]:

def bit_xor(a,b):
    return int(bool(a)^bool(b))

def error_rate(a,b):
    error=np.zeros(len(a))
    for i in range(len(a)):
        error[i]=bit_xor(a[i],b[i])
    return np.sum(error)/len(a)    
    

def bsmc(n,alpha):
    a=np.zeros(n,dtype=np.int)
    a[0]=int(np.random.random()>0.5)

    for i in range(int(n)-1):
        trans=int(np.random.random()<alpha)
        a[i+1]=bit_xor(a[i],trans) 
    return a

def bsc(x,delta):
    z=np.zeros(len(x),dtype=np.int)
    for i in range(len(x)):
        noise=int(np.random.random()<delta)
        z[i]=bit_xor(x[i],noise)
    return z
        
def dude(z,k,delta):
   # print "Running DUDE algorithm"
    n=len(z)
    x_hat=np.zeros(n,dtype=np.int)

    th_0=2*delta*(1-delta)
    m={}
    
    for i in range(k,n-k):
        context=z[i-k:i].tolist()+z[i+1:i+k+1].tolist()
        context_str = ''.join(str(e) for e in context)
        
        if not m.has_key(context_str):
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
   # print "Running DUDE algorithm"
    n=len(z)
    x_hat=np.zeros(n,dtype=np.int)
    s_hat=x_hat.copy()

    th_0=2*delta*(1-delta)
    th_1=delta**2+(1-delta)**2
    
    m={}
    
    for i in range(k,n-k):
        context=z[i-k:i].tolist()+z[i+1:i+k+1].tolist()
        context_str = ''.join(str(e) for e in context)
        
        if not m.has_key(context_str):
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
    n=len(z)
    x_hat=z.copy()
    for i in range(k,n-k):
        if s[i]==0:
            x_hat[i]=z[i]
        elif s[i]==1:
            x_hat[i]=0
        else:
            x_hat[i]=1
    return x_hat


def make_data_for_ndude(Z,k,L,nb_classes,n):
    c_length=2*k
    C=np.zeros((n-2*k, 2*k*nb_classes))

    for i in range(k,n-k):
        c_i = vstack((Z[i-k:i,],Z[i+1:i+k+1,])).reshape(1,2*k*nb_classes)
        C[i-k,]=c_i
        
    Y=dot(Z[k:n-k,],L)    
    return C,Y

def make_binary_image(im):
    im_bin=im.copy()
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i,j]>127: 
                im_bin[i,j]=1
            else:
                im_bin[i,j]=0
    
    return im_bin

# In[7]:

#n=10000
#print n
#alpha=0.1
#delta=0.1
#
#x=bsmc(n,alpha)
## print x
#
#z=bsc(x,delta)
## print z
#
#error = error_rate(x,z)
#print error
#
#err_k=np.zeros(6)
#for k in range(0,6):
#    x_hat= dude(z,k,delta)
#
#    error =error_rate(x,x_hat)
#    err_k[k]=error
#    print error
#    
#k=range(0,6)
#plt.plot(k,err_k)
# print m['1111']
# print m['1111'][0]
# print np.sum(m['1111'])
# print float(m['1111'][0]) / float(np.sum(m['1111']))
        
    



# In[ ]:



