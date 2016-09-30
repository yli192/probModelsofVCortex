#! /usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
#from skimage.color import rgb2gray
from scipy import misc
from scipy.signal import convolve2d
import pickle

def myimshow(I, **kwargs):
    # utility function to show image

    plt.figure();
    plt.axis('off')
    plt.imshow(I, cmap=plt.gray(), **kwargs)

def genSinusoid(sz, A, omega, rho):
    # Generate Sinusoid grating
    # sz: size of generated image (width, height)
    radius = (int(sz[0]/2.0), int(sz[1]/2.0))
    [x, y] = np.meshgrid(range(-radius[0], radius[0]+1), range(-radius[1], radius[1]+1)) # x and y are both 33 x 33 ndarray

    stimuli = A * np.cos(omega[0] * x  + omega[1] * y + rho)
    return stimuli

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def genGabor(sz, omega, theta, func=np.cos, K=np.pi): #theta controls the filters orientation and omega is the frequency
    radius = (int(sz[0]/2.0), int(sz[1]/2.0))
    [x, y] = np.meshgrid(range(-radius[0], radius[0]+1), range(-radius[1], radius[1]+1))

    x1 = x * np.cos(theta) + y * np.sin(theta)
    y1 = -x * np.sin(theta) + y * np.cos(theta)

    gauss = omega**2 / (4*np.pi * K**2) * np.exp(- omega**2 / (8*K**2) * ( 4 * x1**2 + y1**2))
    #myimshow(gauss)
    sinusoid = func(omega * x1) * np.exp(K**2 / 2)
    #myimshow(sinusoid)
    gabor = gauss * sinusoid
    return gabor

#gen Gabor filter banks
theta = np.arange(0, np.pi, np.pi/4) # range of theta
omega = np.arange(0.2, 0.6, 0.1) # range of omega: 0.2 0.3 0.4 0.5
params = [(t,o) for o in omega for t in theta] #returns combinations of theta and omega
sinFilterBank = []
cosFilterBank = []
gaborParams = []
for (theta, omega) in params:
    #print theta,omega
    gaborParam = {'omega':omega, 'theta':theta, 'sz':(128, 128)}
    sinGabor = genGabor(func=np.sin, **gaborParam)
    cosGabor = genGabor(func=np.cos, **gaborParam)
    sinFilterBank.append(sinGabor)
    cosFilterBank.append(cosGabor)
    gaborParams.append(gaborParam)

plt.figure()
n = len(sinFilterBank)
for i in range(n):
    plt.subplot(4,4,i+1)
    # title(r'$\theta$={theta:.2f}$\omega$={omega}'.format(**gaborParams[i]))
    plt.axis('off'); plt.imshow(sinFilterBank[i],cmap=plt.gray())

plt.figure()
for i in range(n):
    plt.subplot(4,4,i+1)
    # title(r'$\theta$={theta:.2f}$\omega$={omega}'.format(**gaborParams[i]))
    plt.axis('off'); plt.imshow(cosFilterBank[i],cmap=plt.gray())


zebra_RGB = misc.imread('/Users/Gary/Desktop/cs485/hw1/data/gabor/Zebra_running_Ngorongoro.jpg')
zebra_gray = rgb2gray(zebra_RGB)
myimshow(zebra_gray)
#pick out filters to try
filter_seqnum_ver = [0,4,8,12]
filter_seqnum_ho = [2 ,6 ,10, 14]
res_all = []

for i in filter_seqnum_ho:
    sinGabor = sinFilterBank[i]
    cosGabor = cosFilterBank[i]
    res_sin = convolve2d(zebra_gray, sinGabor, mode='valid')
    res_cos = convolve2d(zebra_gray, cosGabor, mode='valid')
    res_all.append(res_sin)
    res_all.append(res_cos)
plt.figure()
n = len(filter_seqnum_ver)

for i in range(2*n):
    plt.subplot(4,2,i+1)
    # title(r'$\theta$={theta:.2f}$\omega$={omega}'.format(**gaborParams[i]))
    plt.axis('off'); plt.imshow(res_all[i],cmap=plt.gray())
#sinGabor = sinFilterBank[14]
#plt.figure(); myimshow(sinGabor)
#res = convolve2d(zebra_gray, sinGabor, mode='valid') # Will take about one minute
#plt.figure(); myimshow(res); # title('response') Book figure

plt.show()

#Quadrature pair

"""theta = np.pi/4
sinGabor = genGabor((129,129), 0.4, theta, np.sin)
cosGabor = genGabor((129,129), 0.4, theta, np.cos)
plt.figure();
plt.subplot(121); plt.axis('off'); plt.imshow(sinGabor, vmin=-0.2, vmax=0.2)
plt.subplot(122); plt.axis('off'); plt.imshow(cosGabor, vmin=-0.2, vmax=0.2)

theta = np.pi/4 + np.pi
sinusoid = genSinusoid((256,256), 1, (omega*np.sin(theta), omega*np.cos(theta)), 0)
plt.figure(); myimshow(sinusoid); plt.title('Stimuli')

response = convolve2d(sinusoid, sinGabor, mode='valid')
response2 = convolve2d(sinusoid, cosGabor, mode='valid')

plt.figure();

plt.subplot(121); plt.imshow(response, vmin=0); plt.title('Response of sin gabor(simple cell)')
plt.subplot(122); plt.imshow(response**2 + response2**2, vmin=0); plt.title('Resp. of complex cell')
unknownGabor = pickle.load(open('/Users/Gary/Desktop/cs485/hw1/data/gabor/unknownGabor.data', 'rb'))
plt.figure(); myimshow(unknownGabor)"""
