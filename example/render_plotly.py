import numpy as np
import plotly.graph_objects as go

# Generate initial points
num_points = 50
frames = 50  # Number of animation frames
x = np.random.rand(num_points, frames)
y = np.random.rand(num_points, frames)
z = np.random.rand(num_points, frames)

# Create figure
fig = go.Figure(
    data=[go.Scatter3d(x=x[:, 0], y=y[:, 0], z=z[:, 0], mode='markers')],
    layout=go.Layout(updatemenus=[dict(type='buttons', showactive=False,
        buttons=[dict(label='Play', method='animate', args=[None])])])
)

# Add frames
fig.frames = [go.Frame(data=[go.Scatter3d(x=x[:, i], y=y[:, i], z=z[:, i], mode='markers')]) for i in range(frames)]

fig.show()

