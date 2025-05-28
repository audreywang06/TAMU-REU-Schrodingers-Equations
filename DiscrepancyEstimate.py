import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

alpha = np.sqrt(2)

def hsum(N, H):
    hlist = np.arange(1,H+1)
    hseries = (1/hlist**2)*(1-np.cos(2*np.pi*hlist*N*alpha))/(1-np.cos(2*np.pi*hlist*alpha))
    return np.sum(hseries)

hsum = np.vectorize(hsum)

N = 100000
H = 1000
Nlist = np.arange(1,N+1)
plt.plot(Nlist, hsum(Nlist, H))
plt.plot(Nlist, 1.2*np.log(Nlist))
plt.show()