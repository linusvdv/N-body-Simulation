import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


filename = "out.xyz"
file = open(filename, "r").read().split()

num_particles = int(file[0])
num_timesteps = int(file[1])
num_timesteps_snapshot = int(file[2])
print(num_particles, num_timesteps, num_timesteps_snapshot)

x = []
y = []
z = []
for i in range(num_particles):
    x.append(float(file[3+i*3]))
    y.append(float(file[3+i*3+1]))
    z.append(float(file[3+i*3+2]))

# Create figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-600, 600)
ax.set_ylim(-600, 600)
ax.set_zlim(-600, 600)
sc = ax.scatter(x, y, z)

def update(frame):
    global x, y, z
    x = []
    y = []
    z = []
    for i in range(num_particles):
        x.append(float(file[3+i*3 + 3*num_particles * frame]))
        y.append(float(file[3+i*3+1 + 3*num_particles * frame]))
        z.append(float(file[3+i*3+2 + 3*num_particles * frame]))
    sc._offsets3d = (x, y, z)
    return sc,

ani = animation.FuncAnimation(fig, update, frames=int(num_timesteps/num_timesteps_snapshot), interval=50, blit=False)
plt.show()
