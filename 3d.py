import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# If you’re in Jupyter, uncomment:
# %matplotlib notebook

# Parameters
a, b, c = np.sqrt(2), np.sqrt(3), np.pi
N = 500  # fewer points to start
n = np.arange(1, N+1)

# Precompute coords
X = (n * a) % 1
Y = (n * b) % 1
Z = (n * c) % 1

# Set up figure + 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,1)
ax.set_xlabel('n·a mod 1')
ax.set_ylabel('n·b mod 1')
ax.set_zlabel('n·c mod 1')

# Initial empty scatter
scat = ax.scatter([], [], [], s=5)

def init():
    scat._offsets3d = ([], [], [])
    return scat,

def animate(i):
    # show up to point i
    scat._offsets3d = (X[:i], Y[:i], Z[:i])
    return scat,

ani = FuncAnimation(
    fig, animate,
    init_func=init,
    frames=N,
    interval=20,
    blit=False,   # ← must be False for 3D
    repeat=False
)

plt.show()
