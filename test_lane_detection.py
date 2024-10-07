from lane_detection import LaneDetection
import matplotlib.pyplot as plt
import numpy as np
import pyglet
from pyglet import gl
from pyglet.window import key
import cv2
import random

# action variables

#########################
# init carla environement
#########################
import sys
import glob
import os
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

client = carla.Client('localhost', 2000)
world = client.get_world()

# spawn vehicle
ego_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
ego_bp.set_attribute('role_name','ego')

ego_color = random.choice(ego_bp.get_attribute('color').recommended_values)
ego_bp.set_attribute('color',ego_color)

spawn_points = world.get_map().get_spawn_points()
number_of_spawn_points = len(spawn_points)

# Find spawn point and spawn
if 0 < number_of_spawn_points:
    random.shuffle(spawn_points)
    ego_transform = spawn_points[0]
    ego_vehicle = world.spawn_actor(ego_bp,ego_transform)
    print('\nEgo is spawned')
else:
    logging.warning('Could not find any spawn points')

# Save image to global variable
im = None
def process_img(image):
    img = np.array(image.raw_data)
    img = img.reshape((240, 320, 4))
    img = img[:, :, :3]
    global im
    im = img

# RGB front facing camera
cam_bp = None
cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
cam_bp.set_attribute('image_size_x',str(320))
cam_bp.set_attribute('image_size_y',str(240))
cam_bp.set_attribute('fov',str(90))
cam_location = carla.Location(0,0,2) # On the front hood
cam_rotation = carla.Rotation(0,0,0) # Facing forward
cam_transform = carla.Transform(cam_location, cam_rotation)
ego_cam = world.spawn_actor(cam_bp,
        cam_transform,
        attach_to=ego_vehicle,
        attachment_type=carla.AttachmentType.Rigid)
ego_cam.listen(process_img)

##################


# define variables
steps = 0

# init modules of the pipeline
LD_module = LaneDetection()

# init extra plot
fig = plt.figure()
plt.ion()
plt.show()

ego_vehicle.set_autopilot(True)
try:
    while True:
        # perform step
        world_snapshot = world.wait_for_tick()

        # lane detection
        splines = LD_module.lane_detection(im)
        
        # outputs during training
        if steps % 2 == 0:
            cv2.imshow('frame', im)
            LD_module.plot_state_lane(im, steps, fig)
            cv2.waitKey(0)

        steps += 1

        # check if stop
except KeyboardInterrupt:
   pass 
