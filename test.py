# Setup

from lane_detection import LaneDetection
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from waypoint_prediction import waypoint_prediction, target_speed_prediction

####################
# Test LaneDetection
####################
ld = LaneDetection()
front_img = cv2.imread('./output/test/front_img1.jpg')
#cv2.imshow("front view", front_img)
#cv2.waitKey(0)

# front2bev
bev = ld.front2bev(front_img)

#cv2.imshow("BEV", bev)
#cv2.waitKey(0)

# gray scale
gray = ld.cut_gray(front_img)

#cv2.imshow("grayscale", gray)
#cv2.waitKey(0)

# edge detection
edge = ld.edge_detection(gray)
#cv2.imshow("edges", edge)
#cv2.waitKey(0)

# Find maxima gradient
maxima_gradient = ld.find_maxima_gradient_rowwise(edge)

# lane_detection
lane_boundary1, lane_boundary2 = ld.lane_detection(front_img)

####################
# Test Path Planning
####################

way_points = waypoint_prediction(lane_boundary1, lane_boundary2, num_waypoints=6, way_type="smooth")
print('Target speed: ', target_speed_prediction(way_points))
t = np.linspace(0, 1, 6)
lane_boundary1_points = np.array(splev(t, lane_boundary1))
lane_boundary2_points = np.array(splev(t, lane_boundary2))

fig, ax = plt.subplots()
ax.plot(lane_boundary1_points[0], lane_boundary1_points[1], color='orange')
ax.plot(lane_boundary2_points[0], lane_boundary2_points[1], color='orange')
ax.plot(way_points[0], way_points[1], 'o', color='blue')
plt.show()
