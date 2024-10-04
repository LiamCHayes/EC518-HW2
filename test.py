# Setup

from lane_detection import LaneDetection
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.pyplot as plt

####################
# Test LaneDetection
####################
ld = LaneDetection()
front_img = cv2.imread('./output/test/front_img1.jpg')
cv2.imshow("front view", front_img)
cv2.waitKey(0)

# front2bev
bev = ld.front2bev(front_img)

cv2.imshow("BEV", bev)
cv2.waitKey(0)

# gray scale
gray = ld.cut_gray(front_img)

cv2.imshow("grayscale", gray)
cv2.waitKey(0)

# edge detection
edge = ld.edge_detection(gray)
cv2.imshow("edges", edge)
cv2.waitKey(0)

# Find maxima gradient
maxima_gradient = ld.find_maxima_gradient_rowwise(edge)

# lane_detection
lane_boundary1, lane_boundary2 = ld.lane_detection(front_img)
fig, ax = plt.subplots()
ax.plot(lane_boundary1[0], lane_boundary1[1])
ax.plot(lane_boundary2[0], lane_boundary2[1])
plt.title("Lane spline interpolation")
plt.show()

