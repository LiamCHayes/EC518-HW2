from lane_detection import LaneDetection
import matplotlib.pyplot as plt
import numpy as np
import pyglet
from pyglet import gl
from pyglet.window import key

# action variables

# init carla environement

# define variables
steps = 0

# init modules of the pipeline
LD_module = LaneDetection()

# init extra plot
fig = plt.figure()
plt.ion()
plt.show()

while True:
    # perform step

    # lane detection
    splines = LD_module.lane_detection(s)
    
    # outputs during training
    if steps % 2 == 0:
        print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
        LD_module.plot_state_lane(s, steps, fig)
    steps += 1

    # check if stop
