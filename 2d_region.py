import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

n = 200

a_seq = [10**(0*k) for k in np.arange(1,n+1)]
p = [1,0,1]
q = [0,1,0]

for i in range(n):
    p[0] = a_seq[i]*p[1] + p[2]
    q[0] = a_seq[i]*q[1] + q[2]

    p[2] = p[1]
    p[1] = p[0]

    q[2] = q[1]
    q[1] = q[0]

a = p[0] / q[0]
b = (1 + np.sqrt(5)) / 2

N = 500
sequence_x = (np.arange(1, N+1) * a) % 1
sequence_y = (np.arange(1, N+1) * b) % 1

# Region bounds
x0, x1 = 0.3, 0.7
y0, y1 = 0.4, 0.8

fig, ax = plt.subplots()
points, = ax.plot([], [], 'o', color="red", markersize=3)
region = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1, edgecolor='blue', facecolor='blue', alpha=0.2)
ax.add_patch(region)

text_total = ax.text(0.02, 0.97, '', transform=ax.transAxes, verticalalignment='top')
text_inside = ax.text(0.02, 0.91, '', transform=ax.transAxes, verticalalignment='top')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

def init():
    points.set_data([], [])
    text_total.set_text('')
    text_inside.set_text('')
    return points, text_total, text_inside

def update(frame):
    x = sequence_x[:frame]
    y = sequence_y[:frame]
    points.set_data(x, y)

    inside = ((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1))
    count_inside = np.sum(inside)

    text_total.set_text(f'Total: {frame}')
    text_inside.set_text(f'In Region: {count_inside}')
    return points, text_total, text_inside

ani = FuncAnimation(fig, update, frames=N+1, init_func=init, blit=True, interval=20)
ani.save("kronecker_2d_with_region.mp4", writer=FFMpegWriter(fps=30))
# plt.show()
