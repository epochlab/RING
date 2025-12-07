# #!/usr/bin/env python3

# bokeh serve --show run_interactive.py

import numpy as np

from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, Slider, Button, Spacer, Select
from bokeh.layouts import column, row

from model import NN

np.set_printoptions(linewidth=np.inf, suppress=True, precision=6)

def update():
    net.step(q.value, u.value, c.value, v.value, delta.value, sigma.value, n_opt.value, theta.value, offset.value, shape.value)

    src_z.data["y"] = net.z
    src_X.data["y"] = net.X
    src_dW.data["y"] = net.dW

    global count
    count += 1
    plot.title.text = f"Touretzky Ring Attractor| N={net.N} | Timestep: {count}"

def reset():
    global net, count
    net = NN(N)
    src_z.data = dict(x=x, y=net.z)
    src_X.data = dict(x=x, y=net.X)
    src_initCond.data = dict(x=x, y=net.z.copy())
    count = 0

# DEFINE MODEL
N = 120
net = NN(N)

x = np.arange(N)
src_z = ColumnDataSource(data=dict(x=x, y=net.z))
src_X = ColumnDataSource(data=dict(x=x, y=net.X))
src_dW = ColumnDataSource(data=dict(x=x, y=net.dW))
src_initCond = ColumnDataSource(data=dict(x=x, y=net.z.copy()))
src_weight = ColumnDataSource(data=dict(x=x, y=net.wij[N//2, :]))
count = 0

# PLOT
plot = figure(width=1240, height=640, x_range=(0, N-1), y_range=(-0.05, 1.5))
plot.line('x', 'y', source=src_z, legend_label="Activity (z)")
plot.scatter('x', 'y', source=src_z, marker="circle", size=6, fill_color="white", line_width=0.5)
plot.line('x', 'y', source=src_X, line_color="red", line_dash="dashed", line_width=0.5, legend_label="External Input (x)")
plot.line('x', 'y', source=src_dW, line_width=0.5, legend_label="Noise (dW)")
plot.line('x', 'y', source=src_initCond, line_color="orange", line_alpha=0.5, line_width=0.5, legend_label="Initial Condition")
plot.line('x', 'y', source=src_weight, line_color="green", line_alpha=0.5, line_width=0.5, legend_label="Weight Profile (centre)")
plot.hbar(y=1.0, left=0, right=N, height=0.001, color="black", alpha=0.1)
plot.xaxis.axis_label = "Neuron Index (N)"
plot.yaxis.axis_label = "Response"
plot.legend.title = "Network Components"
plot.legend.location = "top_left"
plot.legend.click_policy = "hide"

# UI PARAMETERS
q = Slider(start=0.0, end=10.0, value=6.0, step=0.1, title="Pooled Inhibition Coefficient (q)")
u = Slider(start=0.0, end=1.0, value=0.5, step=0.01, title="Pooled Inhibition Strength (u)")
c = Slider(start=0.0, end=10.0, value=1.0, step=0.1, title="Noise (c)")
n_opt = Slider(start=1.0, end=10, value=2, step=1, title="No. Options (N)")
v = Slider(start=0.0, end=5.0, value=0.2, step=0.01, title="External Input (X)")
delta = Slider(start=0.0, end=1.0, value=1.0, step=0.1, title="Xin Delta (ΔX)")
sigma = Slider(start=0.01, end=10.0, value=8.0, step=0.01, title="Input Sigma (σ)")
theta = Slider(start=0, end=N, value=80, step=2, title="Distance (𝜃)")
offset = Slider(start=0, end=120, value=0, step=1, title="Offset")

shape = Select(
    title="Input Shape",
    value="gaussian",
    options=["gaussian", "triangle", "laplace"],
    sizing_mode = "stretch_width"
)

reset_button = Button(label="Reset", button_type="success")
reset_button.on_click(reset)

layout = column(plot,
    row(column(Spacer(height=10), q, u, c, reset_button),
        Spacer(width=50),
        column(Spacer(height=10), shape, Spacer(height=5), n_opt, v, delta, sigma, theta, offset),
        align="center")
)

# RENDER
curdoc().add_root(layout)
curdoc().add_periodic_callback(update, 10)