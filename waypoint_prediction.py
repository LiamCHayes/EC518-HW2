import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time


def normalize(v):
    norm = np.linalg.norm(v,axis=0) + 0.00001
    return v / norm.reshape(1, v.shape[1])

def curvature(waypoints):
    '''
    ##### TODO #####
    Curvature as the sum of the normalized dot product between the way elements
    Implement second term of the smoothin objective.

    args: 
        waypoints [2, num_waypoints] !!!!!
    '''
    curvature = 0
    for point in range(waypoints.shape[1]-1):
        numerator = (waypoints[:, point+1] - waypoints[:, point]) * (waypoints[:, point] - waypoints[:, point+1])
        numerator = numerator.sum()
        denominator = np.sqrt((waypoints[:, point+1] - waypoints[:, point]).dot((waypoints[:, point+1] - waypoints[:, point]))) * np.sqrt((waypoints[:, point] - waypoints[:, point+1]).dot((waypoints[:, point] - waypoints[:, point+1])))
        curvature = curvature + numerator/denominator

    return curvature


def smoothing_objective(waypoints, waypoints_center, weight_curvature=40):
    '''
    Objective for path smoothing

    args:
        waypoints [2 * num_waypoints] !!!!!
        waypoints_center [2 * num_waypoints] !!!!!
        weight_curvature (default=40)
    '''
    waypoints = waypoints.reshape(2, -1)
    # mean least square error between waypoint and way point center
    ls_tocenter = np.mean((waypoints_center - waypoints)**2)

    # derive curvature
    curv = curvature(waypoints.reshape(2,-1))

    return -1 * weight_curvature * curv + ls_tocenter


def waypoint_prediction(roadside1_spline, roadside2_spline, num_waypoints=6, way_type = "smooth"):
    '''
    ##### TODO #####
    Predict waypoint via two different methods:
    - center
    - smooth 

    args:
        roadside1_spline
        roadside2_spline
        num_waypoints (default=6)
        parameter_bound_waypoints (default=1)
        waytype (default="smoothed")
    '''
    if way_type == "center":
        t = np.linspace(0, 1, num_waypoints)
        lane_boundary1_points = np.array(splev(t, roadside1_spline))
        lane_boundary2_points = np.array(splev(t, roadside2_spline))
        
        # derive center between corresponding roadside points
        way_points = np.empty((2, num_waypoints))
        for i in range(num_waypoints):
            way_points[0,i] = (lane_boundary1_points[0,i] + lane_boundary2_points[0,i])/2
            way_points[1,i] = (lane_boundary1_points[1,i] + lane_boundary2_points[1,i])/2

        # output way_points with shape(2 x Num_waypoints)
        return way_points
    
    elif way_type == "smooth":
        t = np.linspace(0, 1, num_waypoints)
        lane_boundary1_points = np.array(splev(t, roadside1_spline))
        lane_boundary2_points = np.array(splev(t, roadside2_spline))
        
        # derive center between corresponding roadside points
        way_points_center = np.empty((2, num_waypoints))
        for i in range(num_waypoints):
            way_points_center[0,i] = (lane_boundary1_points[0,i] + lane_boundary2_points[0,i])/2
            way_points_center[1,i] = (lane_boundary1_points[1,i] + lane_boundary2_points[1,i])/2

        # optimization
        way_points = minimize(smoothing_objective, 
                      (way_points_center), 
                      args=way_points_center)["x"]

        return way_points.reshape(2,-1)


def target_speed_prediction(waypoints, num_waypoints_used=5,
                            max_speed=60, exp_constant=4.5, offset_speed=30):
    '''
    ##### TODO #####
    Predict target speed given waypoints
    Implement the function using curvature()

    args:
        waypoints [2,num_waypoints]
        num_waypoints_used (default=5)
        max_speed (default=60)
        exp_constant (default=4.5)
        offset_speed (default=30)
    
    output:
        target_speed (float)
    '''
    target_speed = (max_speed - offset_speed) * np.exp(-exp_constant * abs(num_waypoints_used - 2 - curvature(waypoints))) + offset_speed 
    
    return target_speed
