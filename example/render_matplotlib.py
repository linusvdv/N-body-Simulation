import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Generate initial positions
num_points = 50
x = np.random.rand(num_points)
y = np.random.rand(num_points)
z = np.random.rand(num_points)

# Create figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x, y, z)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)

# Update function
def update(frame):
    global x, y, z
    x += (np.random.rand(num_points) - 0.5) * 0.1
    y += (np.random.rand(num_points) - 0.5) * 0.1
    z += (np.random.rand(num_points) - 0.5) * 0.1
    sc._offsets3d = (x, y, z)  # Update positions
    return sc,

# Animate
ani = animation.FuncAnimation(fig, update, frames=100, interval=50, blit=False)
plt.show()

