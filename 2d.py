import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

n = 200

a = [10**(0*k) for k in np.arange(1,n+1)]

p = [1,0,1]
q = [0,1,0]

for i in range(n):
    p[0]=a[i]*p[1]+p[2]
    q[0]=a[i]*q[1]+q[2]

    p[2] = p[1]
    p[1] = p[0]

    q[2] = q[1]
    q[1] = q[0]

    print(p,q)

print(p[0],q[0])

a = p[0]/q[0]
print(a)

b = (1+np.sqrt(5))/2 # (1+np.sqrt(7))/2 # np.sqrt(3) # Irrational number
N = 100000 # Maximum n

sequence_x = (np.arange(1,N+1)*a) % 1
sequence_y = (np.arange(1,N+1)*b) % 1

fig, ax = plt.subplots()
line, = ax.plot([], [], 'o', color="red")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

def init():
    line.set_data([], [])
    return line,

def update(frame):
    x = sequence_x[:frame]
    y = sequence_y[:frame]
    line.set_data(x, y)
    return line,

ani = animation.FuncAnimation(fig, update, frames=N+1, init_func=init, blit=True, interval=0.01)

plt.show()

def prop(M, d_1, d_2):
    d_1 = 0.5 # in [0,b)
    d_2 = 0.5 # in [0,b)
    count = np.zeros(M)
    for i in range(M):
        count[i] = (np.sum((sequence_x[:M] >= 0) & (sequence_x[:M] < d_1) & (sequence_y[:M] >= 0) & (sequence_y[:M] < d_2)))/(i+1)
    return count

nlist = np.arange(1,N+1)

plt.plot(nlist, prop(N, 0.5, 0.5))
plt.show()
