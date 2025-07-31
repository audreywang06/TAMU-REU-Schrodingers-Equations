import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.cm import hsv
from matplotlib.colors import Normalize
from scipy.sparse import diags
from scipy.sparse.linalg import expm

# --- Adjustable Parameters ---
N = 200                     # number of lattice sites
plot_range = (80, 160)      # which region to show in the animation
barrier_start = 140         # index where the barrier starts
barrier_width = 10          # width of the potential barrier
barrier_height = 0.5          # height of the potential barrier
wavepacket_center = 100     # start wavepacket near the right (to go leftward)
wavepacket_width = 5.0      # spatial width of the Gaussian
wavepacket_momentum =  1    # negative = left-moving
timesteps = 150             # total time steps in animation
dt = 0.5                   # time step size
output_file = "Barrier_Simulation/discrete_wavepacket_reflection.gif"

# --- Create potential ---
x = np.arange(N)
V = np.zeros(N)
V[barrier_start:barrier_start+barrier_width] = barrier_height

# --- Initial wavefunction: Gaussian with complex momentum ---
psi0 = np.exp(-0.5 * ((x - wavepacket_center)/wavepacket_width)**2) * np.exp(1j * wavepacket_momentum * x) # + np.exp(-0.5 * ((x - (wavepacket_center + 40))/wavepacket_width)**2) * np.exp(1j * -wavepacket_momentum * x)
psi0 /= np.linalg.norm(psi0)

# --- Hamiltonian (tight-binding Laplacian + potential) ---
main_diag = V + 1.0
off_diag = -0.5 * np.ones(N - 1)
H = diags([main_diag, off_diag, off_diag], [0, -1, 1], format='csc')

# --- Exact time evolution operator: U = exp(-i H dt) ---
U = expm(-1j * H * dt)

# --- Set up figure ---
fig, ax = plt.subplots(figsize=(10, 4))
ax.set_xlim(plot_range[0], plot_range[1])
ax.set_ylim(0, 0.8)
# ax.set_title("Wavefunction Time Evolution",fontsize=16)
ax.set_xlabel("$n$",fontsize=14)
ax.set_ylabel("$\psi(n)$",fontsize=14)

bars = ax.bar(x[plot_range[0]:plot_range[1]],
              np.abs(psi0[plot_range[0]:plot_range[1]]),
              color=hsv((np.angle(psi0[plot_range[0]:plot_range[1]]) + np.pi)/(2 * np.pi)),
              width=1.0, align='center')

# Plot potential barrier
ax.bar(x[plot_range[0]:plot_range[1]],
       V[plot_range[0]:plot_range[1]] / barrier_height,
       color='gray', alpha=0.2, width=1.0)

# --- Normalize phase color map ---
norm = Normalize(-np.pi, np.pi)
psi = psi0.copy()

# --- Animation update function ---
def update(frame):
    global psi
    psi = U @ psi
    mags = np.abs(psi[plot_range[0]:plot_range[1]])
    phases = np.angle(psi[plot_range[0]:plot_range[1]])
    colors = hsv(norm(phases))
    for i, bar in enumerate(bars):
        bar.set_height(mags[i])
        bar.set_color(colors[i])
    return bars

# --- Run animation ---
ani = animation.FuncAnimation(fig, update, frames=timesteps, interval=30, blit=False)

# --- Save as GIF ---
ani.save(output_file, writer='pillow', fps=60)