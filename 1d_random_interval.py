import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

# Constants
a = np.sqrt(2)
n_values = np.arange(1, 1001)

# Calculate na mod 1
na_mod_1 = (n_values * a) % 1

# Function to generate a random interval
interval_start = random.uniform(0, 0.9)  # Ensure the interval can fit within [0, 1]
interval_end = interval_start + random.uniform(0.05, 0.1)  # Random interval length

# Count how many points fall within the interval
def count_points_in_interval():
    return np.sum((na_mod_1 >= interval_start) & (na_mod_1 <= interval_end))

# Set up the plot for a number line from 0 to 1
fig, ax = plt.subplots()
ax.set_xlim(0, 1)  # Set x-limits from 0 to 1
ax.set_ylim(-0.05, 0.05)  # Narrow y-limits to simulate a number line
ax.set_xlabel('na mod 1')
ax.get_yaxis().set_visible(False)  # Hide the y-axis
ax.set_title('plot')

# Plot each point one by one on the number line
line, = ax.plot([], [], 'bo')

# Plot the random interval
interval_line = ax.plot([interval_start, interval_end], [0, 0], 'r-', linewidth=2)[0]

# Initialization function
def init():
    line.set_data([], [])
    interval_line.set_data([interval_start, interval_end], [0, 0])
    return line, interval_line

# Animation function
def animate(i):
    line.set_data(na_mod_1[:i], np.zeros(i))  # All y-values are zero
    return line, interval_line

# Count and print the number of points in the interval
points_in_interval = count_points_in_interval()
print(f'Number of points in the interval [{interval_start:.2f}, {interval_end:.2f}]: {points_in_interval}')

# Create animation
ani = FuncAnimation(fig, animate, init_func=init, frames=len(n_values), interval=10, blit=True, repeat=False)

# Show plot
plt.show() 
