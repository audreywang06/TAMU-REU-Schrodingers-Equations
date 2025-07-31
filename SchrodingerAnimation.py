import numpy as np
import matplotlib.pyplot as plt
import imageio
from matplotlib.gridspec import GridSpec
from io import BytesIO
from scipy.linalg import expm
import random as rand

# --- Parameters ---
N_sim = 1000                     # Total number of lattice sites for simulation
N_plot = 30                     # Number of central sites to plot
T = 100                          # Number of time steps
dt = 1                         # Time step size
p_list = [1]                    # Moments to compute
x_sim = np.arange(-N_sim//2, N_sim//2)  # Simulation lattice positions
x_plot = np.arange(-N_plot//2, N_plot//2)  # Plotting positions

# --- Define potential function ---
alpha = (np.sqrt(5)-1)/2
def potential(x):
    return 2*np.cos(2*alpha*x*np.pi)
    # return 0.01 * x**2         # Harmonic oscillator
    # return np.zeros_like(x)               # Free particle
    # return 10 * (np.abs(x) > 10)          # Finite square well
    # return np.random.normal(0, 1, len(x)) # Random disorder

V_sim = potential(x_sim)

# --- Initial wavefunction: Gaussian wave packet ---
def psi0(x):
    return np.exp(-0.5 * (x / 3)**2) # * np.exp(1j * x * 0.5)

psi = psi0(x_sim)
psi = psi / np.linalg.norm(psi)

# --- Build Hamiltonian with potential ---
H = np.zeros((N_sim, N_sim), dtype=complex)
for i in range(N_sim):
    if i > 0:
        H[i, i - 1] = 1
    if i < N_sim - 1:
        H[i, i + 1] = 1
    H[i, i] = V_sim[i]  # Make sure potential is added here

# --- Time evolution operator ---
U = expm(-1j * dt * H)

# --- Track moments and frames ---
moments = {p: [] for p in p_list}
times = []
frames = []

# --- Index window for plotting center region ---
plot_start = N_sim//2 - N_plot//2
plot_end = plot_start + N_plot

# --- Axis bounds and settings ---
moment_ylim = [0, 70] #(N_plot//2)**p_list[-1] * 0.6]
psi_ylim = [0, 0.5]
bar_width = 0.8

# --- Main loop ---
for t_step in range(T):
    prob_density = np.abs(psi)**2

    # Compute moments
    for p in p_list:
        moments[p].append(np.sum(prob_density * np.abs(x_sim)**p))
    times.append(t_step * dt)

    # Setup figure
    fig = plt.figure(figsize=(10, 4))
    gs = GridSpec(1, 2, width_ratios=[2, 3])

    # --- Left: Wavefunction and potential ---
    ax0 = fig.add_subplot(gs[0])
    n_vals = x_plot
    psi_vals = prob_density[plot_start:plot_end]
    V_vals = V_sim[plot_start:plot_end]

    # Scale potential to match display height
    V_scaled = V_vals*.8 / np.max(V_sim + 1e-9) * max(psi_ylim)

    # Plot potential (background bars)
    ax0.bar(n_vals, V_scaled, color='lightgreen', width=bar_width, align='center', label='Potential')

    # Plot wavefunction (foreground bars)
    ax0.bar(n_vals, psi_vals, color='steelblue', width=bar_width, align='center', label='|ψ(n)|²')

    ax0.set_ylim(psi_ylim)
    ax0.set_title(f'Wavefunction at t = {t_step*dt:.2f}')
    ax0.set_xlabel('n')
    ax0.set_ylabel('Amplitude')
    ax0.legend(loc='upper right')

    # --- Right: Moments over time ---
    ax1 = fig.add_subplot(gs[1])
    for p in p_list:
        ax1.plot(times, moments[p], label=f'p={p}')
    ax1.set_xlim([0, T * dt])
    ax1.set_ylim(moment_ylim)
    ax1.set_xlabel('Time ($t$)')
    ax1.set_ylabel(r'$\langle X_H|_\phi^1\rangle(t)$')
    ax1.set_title('1st Moment Over Time')
    ax1.legend()

    fig.tight_layout()

    # Save frame to memory
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    frames.append(imageio.v2.imread(buf))
    buf.close()
    plt.close()

    # Apply exact time evolution
    psi = U @ psi

# Save as GIF
imageio.mimsave(
    "schrodinger_moments_golden.gif",
    frames,
    duration=0.05,
    # loop=0  # This ensures the GIF will loop forever (auto-repeats in PowerPoint)
)