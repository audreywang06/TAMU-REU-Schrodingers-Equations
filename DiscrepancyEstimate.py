import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

alpha = (1+np.sqrt(5))/2 # np.sqrt(2) #(1+np.sqrt(5))/2
N = 1000

# Define a as a continued fraction

n = 200

a = [10**(2*k) for k in np.arange(1,n+1)]

p = [1,0,1]
q = [0,1,0]

# for i in range(n):
#     p[0]=a[i]*p[1]+p[2]
#     q[0]=a[i]*q[1]+q[2]

#     p[2] = p[1]
#     p[1] = p[0]

#     q[2] = q[1]
#     q[1] = q[0]

# alpha = 1/1000 # p[0]/q[0]

# Define the {na} sequence
sequence = np.arange(1,N+1)*alpha % 1

# Calculate the discrepancy D(N;a) for a specified interval [0,a)
def specific_discrepancy(N_, a_):
  seq = sequence[:N_]
  count = np.sum((seq >= 0) & (seq < a_))
  return abs(count/N_-a_)

# print(specific_discrepancy(N, 0.5))

def discrepancy(N_):
  x0 = sp.optimize.minimize(lambda x: -specific_discrepancy(N_, x), 0.5, bounds=np.array([(0,1)]))
  return specific_discrepancy(N_, x0.x)

discrepancy = np.vectorize(discrepancy)

nlist = np.arange(1, N+1)
plt.plot(nlist, discrepancy(nlist))
# plt.plot(nlist, (5)**(1/3)/nlist**(2/3))
# plt.plot(nlist, 0.4*np.log(nlist)/nlist)
# plt.ylim([0,1])
plt.show()
