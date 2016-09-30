#! /usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from scipy.signal import convolve2d
import pickle
import operator

def myimshow(I, **kwargs):
    # utility function to show image
    plt.figure();
    plt.axis('off')
    plt.imshow(I, cmap=plt.gray(), **kwargs)

def surf(X, Y, Z, **kargs):
    # Plot 3D data as surface, similar to surf(X,Y,Z) of http://www.mathworks.com/help/matlab/ref/surf.html
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, **kargs)

def genSinusoid(sz, A, omega, rho):
    # Generate Sinusoid grating
    # sz: size of generated image (width, height)
    radius = (int(sz[0]/2.0), int(sz[1]/2.0))
    [x, y] = np.meshgrid(range(-radius[0], radius[0]+1), range(-radius[1], radius[1]+1)) # a BUG is fixed in this line

    stimuli = A * np.cos(omega[0] * x  + omega[1] * y + rho)
    return stimuli

unknownLoG = pickle.load(open('/Users/Gary/Desktop/cs485/hw1/data/gabor/unknownLoG.data', 'rb'))
myimshow(unknownLoG)
[X, Y] = np.meshgrid(np.arange(-100, 101), np.arange(-100, 101))
surf(X, Y, unknownLoG, alpha=0.3)


rho_all = np.arange(0.0, 1.6, 0.2)
#rho_all = [0]
omega = np.arange(0.0, 1.6, 0.005)
theta = np.arange(0, np.pi, np.pi/100)
params = [(t,o) for o in omega for t in theta]
res_all = {}
for rho in rho_all:
    for (omega, theta) in params:
        sinusoid = genSinusoid(unknownLoG.shape, 1, (omega*np.cos(theta), omega*np.sin(theta)), rho)
        response = convolve2d(sinusoid, unknownLoG, mode='valid')
        res_all[omega,theta,rho]=response[0][0]
#myimshow(sinusoid)
sorted_res_all = sorted(res_all.items(), key=operator.itemgetter(1))
print 'Strength of response:', sorted_res_all, sorted_res_all[-1][0][0]



rho = np.pi/2
omega =  sorted_res_all[-1][0][0]
theta = sorted_res_all[-1][0][1]
sinusoid = genSinusoid((65,65), 1, (omega*np.cos(theta), omega*np.sin(theta)), rho)
myimshow(sinusoid)
plt.show()
