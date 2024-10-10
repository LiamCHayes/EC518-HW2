import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time


class LateralController:
    '''
    Lateral control using the Stanley controller

    functions:
        stanley 

    init:
        gain_constant (default=5)
        damping_constant (default=0.5)
    '''


    def __init__(self, gain_constant=5, damping_constant=0.6):

        self.gain_constant = gain_constant
        self.damping_constant = damping_constant
        self.previous_steering_angle = 0


    def stanley(self, waypoints, speed):
        '''
        ##### TODO #####
        one step of the stanley controller with damping
        args:
            waypoints (np.array) [2, num_waypoints]
            speed (float)
        '''
        # derive orientation error as the angle of the first path segment to the car orientation 
        car = np.array([160, 1])
        way = waypoints[:, 1]
        orient_err = np.arccos(np.dot(car, way) / (np.linalg.norm(car) * np.linalg.norm(way)))

        # derive cross track error as distance between desired waypoint at spline parameter equal zero ot the car position
        cross_err = 160 - waypoints[0, 0] 

        # derive stanley control law
        # prevent division by zero by adding as small epsilon
        k = 1
        steer = orient_err + np.arctan((k*cross_err)/speed)

        # derive damping term
        D = 0.2
        steering_angle = steer - D * (steer - self.previous_steering_angle)
        self.previous_steering_angle = steering_angle
        # clip to the maximum stering angle (0.4) and rescale the steering action space
        return np.clip(steering_angle, -0.4, 0.4) / 0.4






