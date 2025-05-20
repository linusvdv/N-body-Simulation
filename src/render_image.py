import numpy as np
import matplotlib.pyplot as plt

# Parameters
filename = "out.xyz"
file = open(filename, "r").read().split()

num_particles = int(file[0])
num_timesteps = int(file[1])
num_timesteps_snapshot = int(file[2])
print(num_particles, num_timesteps, num_timesteps_snapshot)

# Compute total frames (i.e., how many snapshots)
num_frames = num_timesteps // num_timesteps_snapshot

# Initialize position history for each particle
x_history = [[] for _ in range(num_particles)]
y_history = [[] for _ in range(num_particles)]

# Read all positions over time
for frame in range(num_frames):
    base_index = 3 + frame * num_particles * 3
    for i in range(num_particles):
        x = float(file[base_index + i*3]) # - float(file[base_index + 1*3])
        y = float(file[base_index + i*3 + 1]) # - float(file[base_index + 1*3 + 1])
        x_history[i].append(x)
        y_history[i].append(y)

x_history = np.array(x_history)
y_history = np.array(y_history)

# Plot the trajectories
plt.rc('axes', axisbelow=True)
plt.figure(figsize=(8, 8))
planet_ids = {
    "Sun": 10,
    "Mercury": 199,
    "Venus": 299,
    "Moon": 301,
    "Earth": 399,
    "Mars": 499,
    "Jupiter": 599,
    "Saturn": 699,
    "Uranus": 799,
    "Neptune": 899
}
for i, (key, value) in enumerate(planet_ids.items()):
    if i in [0, 3, 4]:
        plt.scatter(x_history[i]-x_history[0], y_history[i]-y_history[0], 1, label=key)

plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("2D Trajectories of the solar system (z-axis ignored)")
plt.axis("equal")
plt.grid(True)
# Optionally show a legend:
plt.legend()
plt.tight_layout()
plt.savefig("trajectories.png", dpi=600)
#plt.show()

