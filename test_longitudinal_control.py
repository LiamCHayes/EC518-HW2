from lane_detection import LaneDetection
from waypoint_prediction import waypoint_prediction, target_speed_prediction
from lateral_control import LateralController
from longitudinal_control import LongitudinalController
import matplotlib.pyplot as plt
import numpy as np
import pyglet
from pyglet import gl
from pyglet.window import key
import cv2
import random

# action variables
a = np.zeros(3)

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
world = client.get_world()

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
LatC_module = LateralController()
LongC_module = LongitudinalController()

# init extra plot
fig = plt.figure()
plt.ion()
plt.show()

c = carla.VehicleControl()
while True:
    # perform step
    world.wait_for_tick()

    if im is not None:
        # lane detection
        lane1, lane2 = LD_module.lane_detection(im)
        speed = ego_vehicle.get_velocity().x

        # waypoint and target_speed prediction
        waypoints = waypoint_prediction(lane1, lane2)
        target_speed = target_speed_prediction(waypoints, max_speed=60, exp_constant=4.5)

        # control
        a[0] = LatC_module.stanley(waypoints, speed)
        a[1], a[2] = LongC_module.control(speed, target_speed)
        c.steer = a[0]
        c.throttle = a[1]
        c.brake = a[2]
        ego_vehicle.apply_control(c)

        # outputs during training
        if steps % 2 == 0:
            print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
            print("speed {:+0.2f} targetspeed {:+0.2f}".format(speed, target_speed))

            LD_module.plot_state_lane(im, steps, fig, waypoints=waypoints)
            #LongC_module.plot_speed(speed, target_speed, steps, fig)
            #cv2.imshow('frame', im)
    steps += 1

    # check if stop
