#! /usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
#from skimage.color import rgb2gray
from scipy import misc
from scipy.signal import convolve2d
import pickle
import operator
def genSinusoid(sz, A, omega, rho):
    # Generate Sinusoid grating
    # sz: size of generated image (width, height)
    radius = (int(sz[0]/2.0), int(sz[1]/2.0))
    [x, y] = np.meshgrid(range(-radius[0], radius[0]+1), range(-radius[1], radius[1]+1)) # x and y are both 33 x 33 ndarray

    stimuli = A * np.cos(omega[0] * x  + omega[1] * y + rho)
    return stimuli

def myimshow(I, **kwargs):
    # utility function to show image
    plt.figure();
    plt.axis('off')
    plt.imshow(I, cmap=plt.gray(), **kwargs)
unknownGabor = pickle.load(open('/Users/Gary/Desktop/cs485/hw1/data/gabor/unknownGabor.data', 'rb'))
myimshow(unknownGabor)
#unknownGabor.astype(float)
# You can use sinusoid as a stimuli
# For example:
rho = 0
omega = np.arange(0.0, 1.6, 0.05)
theta = np.arange(0, np.pi, np.pi/10)
params = [(t,o) for o in omega for t in theta]
res_all = {}
for (omega, theta) in params:
    sinusoid = genSinusoid((65,65), 1, (omega*np.cos(theta), omega*np.sin(theta)), rho)
    response = convolve2d(sinusoid, unknownGabor, mode='valid')
    res_all[omega,theta]=(response[0][0])

#myimshow(sinusoid)
sorted_res_all = sorted(res_all.items(), key=operator.itemgetter(1))
print 'Strength of response:', sorted_res_all, sorted_res_all[-1][0][0]



rho = np.pi/2
omega =  sorted_res_all[-1][0][0]
theta = sorted_res_all[-1][0][1]
sinusoid = genSinusoid((65,65), 1, (omega*np.cos(theta), omega*np.sin(theta)), rho)
myimshow(sinusoid)
plt.show()
