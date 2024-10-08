import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time
import cv2

class LaneDetection:
    '''
    Lane detection module using edge detection and b-spline fitting

    args: 
        cut_size (cut_size=120) cut the image at the front of the car
        spline_smoothness (default=10)
        gradient_threshold (default=14)
        distance_maxima_gradient (default=3)

    '''

    def __init__(self, cut_size=120, spline_smoothness=10, gradient_threshold=14, distance_maxima_gradient=3):
        self.car_position = np.array([160,0])
        self.spline_smoothness = spline_smoothness
        self.cut_size = cut_size
        self.gradient_threshold = gradient_threshold
        self.distance_maxima_gradient = distance_maxima_gradient
        self.lane_boundary1_old = 0
        self.lane_boundary2_old = 0
        self.bev_to_front = None
    

    def front2bev(self, front_view_image):
        '''
        ##### TODO #####
        This function should transform the front view image to bird-eye-view image.

        input:
            front_view_image)320x240x3

        output:
            bev_image 320x240x3

        '''
        # Extract region of interest (bottom to horizon)
        roi = front_view_image[115:240, 0:320]
        img_height = 125 
        img_width = 320

        # Matrix of 4 source points
        src = np.float32([[65, img_height], [305, img_height], [140, 30], [192, 30]])
        # Matrix of 4 destination points
        dst = np.float32([[140, img_height], [192, img_height], [140, 30], [192, 30]])
        
        # Get transformation matrix
        front_to_bev = cv2.getPerspectiveTransform(src, dst)
        self.bev_to_front = cv2.getPerspectiveTransform(dst, src)
        
        # Transform
        bev_image = cv2.warpPerspective(roi, front_to_bev, (img_width, img_height))

        return bev_image 


    def cut_gray(self, state_image_full):
        '''
        ##### TODO #####
        This function should cut the image at the front end of the car
        and translate to grey scale

        input:
            state_image_full 320x240x3

        output:
            gray_state_image 320x120x1

        '''
        # Get BEV
        bev = self.front2bev(state_image_full)

        # Cut off car and resise to 320x120
        bev = bev[0:115, 0:320]
        bev = cv2.resize(bev, (320, 120))
        
        # To gray scale
        gray_state_image = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)

        return gray_state_image[::-1]


    def edge_detection(self, gray_image):
        '''
        ##### TODO #####
        In order to find edges in the gray state image, 
        this function should derive the absolute gradients of the gray state image.
        Derive the absolute gradients using numpy for each pixel. 
        To ignore small gradients, set all gradients below a threshold (self.gradient_threshold) to zero. 

        input:
            gray_state_image 320x120x1

        output:
            gradient_sum 320x120x1

        '''
        # Normalize to make gradients more clear
        gray_image = cv2.normalize(gray_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        Gx = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]])
        Gy = np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]])

        gradient_sum = np.zeros((120, 320))
        # Convolve gradients
        for row in range(1, gray_image.shape[0]-1):
            for col in range(1, gray_image.shape[1]-1):
                Iij = gray_image[row-1:row+2, col-1:col+2]
                grad_x = np.multiply(Iij, Gx).sum()
                grad_y = np.multiply(Iij, Gy).sum()
                grad_mag = (grad_x**2 + grad_y**2)**0.5

                gradient_sum[row, col] = grad_mag if grad_mag > 100 else 0
        
        for row in range(gradient_sum.shape[0]):
            peak, _ = find_peaks(gradient_sum[row, :], distance=self.distance_maxima_gradient) 
            if len(peak) != 0:
                gradient_sum[row, peak] = 255
                gradient_sum[row, -peak] = 0
            else:
                gradient_sum[row, :] = 0

        return gradient_sum


    def find_maxima_gradient_rowwise(self, gradient_sum):
        '''
        ##### TODO #####
        This function should output arguments of local maxima for each row of the gradient image.
        You can use scipy.signal.find_peaks to detect maxima. 
        Hint: Use distance argument for a better robustness.

        input:
            gradient_sum 320x120x1

        output:
            maxima (np.array) 2x Number_maxima

        '''
        argmaxima = np.zeros((2, 1))
        for row in range(gradient_sum.shape[0]):
            edges = np.nonzero(gradient_sum[row, :])[0]

            if len(edges) != 0:
                rowmaxima = np.vstack((edges, np.full(len(edges), row)))
                argmaxima = np.hstack((argmaxima, rowmaxima))
        
        argmaxima = argmaxima[:, 1:]

        return argmaxima


    def find_first_lane_point(self, gradient_sum):
        '''
        Find the first lane_boundaries points above the car.
        Special cases like just detecting one lane_boundary or more than two are considered. 
        Even though there is space for improvement ;) 

        input:
            gradient_sum 320x120x1

        output: 
            lane_boundary1_startpoint
            lane_boundary2_startpoint
            lanes_found  true if lane_boundaries were found
        '''
        
        # Variable if lanes were found or not
        lanes_found = False
        row = 0

        # loop through the rows
        while not lanes_found:
            
            # Find peaks with min distance of at least 3 pixel 
            argmaxima = find_peaks(gradient_sum[row],distance=3)[0]

            # if one lane_boundary is found
            if argmaxima.shape[0] == 1:
                lane_boundary1_startpoint = np.array([[argmaxima[0],  row]])

                if argmaxima[0] < 160:
                    lane_boundary2_startpoint = np.array([[0,  row]])
                else: 
                    lane_boundary2_startpoint = np.array([[320,  row]])

                lanes_found = True
            
            # if 2 lane_boundaries are found
            elif argmaxima.shape[0] == 2:
                lane_boundary1_startpoint = np.array([[argmaxima[0],  row]])
                lane_boundary2_startpoint = np.array([[argmaxima[1],  row]])
                lanes_found = True

            # if more than 2 lane_boundaries are found
            elif argmaxima.shape[0] > 2:
                # if more than two maxima then take the two lanes next to the car, regarding least square
                A = np.argsort((argmaxima - self.car_position[0])**2)
                lane_boundary1_startpoint = np.array([[argmaxima[A[0]], row]])
                lane_boundary2_startpoint = np.array([[argmaxima[A[1]], row]])
                lanes_found = True

            row += 1
            
            # if no lane_boundaries are found
            if row == self.cut_size:
                lane_boundary1_startpoint = np.array([[0,  0]])
                lane_boundary2_startpoint = np.array([[0,  0]])
                break

        return lane_boundary1_startpoint, lane_boundary2_startpoint, lanes_found


    def lane_detection(self, state_image_full):
        '''
        ##### TODO #####
        This function should perform the road detection 

        args:
            state_image_full [320, 240, 3]

        out:
            lane_boundary1 spline
            lane_boundary2 spline
        '''

        # to gray
        gray_state = self.cut_gray(state_image_full)

        # edge detection via gradient sum and thresholding
        gradient_sum = self.edge_detection(gray_state)
        maxima = self.find_maxima_gradient_rowwise(gradient_sum)

        # first lane_boundary points
        lane_boundary1_points, lane_boundary2_points, lane_found = self.find_first_lane_point(gradient_sum)

        # if no lane was found,use lane_boundaries of the preceding step
        if lane_found:
            row_idx = lane_boundary1_points[-1][1]
            for row in range(row_idx+1, gradient_sum.shape[0]-1):
                # Boundary 1
                point = lane_boundary1_points[-1]
                x = point[1]
                y = point[0]
                distances = np.empty(len(np.where(maxima[1, :] == row)[0]))
                for dist_idx, i in enumerate(np.where(maxima[1, :] == row)[0]):
                    x_next = maxima[1, i]
                    y_next = maxima[0, i]
                    distances[dist_idx] = ((x-x_next)**2 + (y-y_next)**2)**0.5
                if len(distances) != 0 and np.min(distances) < 100:
                    lane_boundary1_points = np.vstack((lane_boundary1_points, maxima[:, np.where(maxima[1, :] == row)[0][distances == np.min(distances)][0]]))
                else: 
                    lane_boundary1_points = np.vstack((lane_boundary1_points, np.array([y, x+1])))
                np.delete(maxima, np.where(maxima[1, :] == row)[0][distances == np.min(distances)][0])
                
                # Boundary 2
                point = lane_boundary2_points[-1]
                x = point[1]
                y = point[0]
                distances = np.empty(len(np.where(maxima[1, :] == row)[0]))
                for dist_idx, i in enumerate(np.where(maxima[1, :] == row)[0]):
                    x_next = maxima[1, i]
                    y_next = maxima[0, i]
                    distances[dist_idx] = ((x-x_next)**2 + (y-y_next)**2)**0.5
                if len(distances) != 0 and np.min(distances) < 100:
                    lane_boundary2_points = np.vstack((lane_boundary2_points, maxima[:, np.where(maxima[1, :] == row)[0][distances == np.min(distances)][0]]))
                else: 
                    lane_boundary2_points = np.vstack((lane_boundary2_points, np.array([y, x+1])))
                np.delete(maxima, np.where(maxima[1, :] == row)[0][distances == np.min(distances)][0])

            # fit splines
            if lane_boundary1_points.shape[0] > 4 and lane_boundary2_points.shape[0] > 4:
                lane_boundary1, u = splprep([lane_boundary1_points[:, 0], lane_boundary1_points[:, 1]], s=0)
                lane_boundary2, u = splprep([lane_boundary2_points[:, 0], lane_boundary2_points[:, 1]], s=0)
            else:
                lane_boundary1 = self.lane_boundary1_old
                lane_boundary2 = self.lane_boundary2_old

        else:
            lane_boundary1 = self.lane_boundary1_old
            lane_boundary2 = self.lane_boundary2_old

        self.lane_boundary1_old = lane_boundary1
        self.lane_boundary2_old = lane_boundary2

        # output the spline
        return lane_boundary1, lane_boundary2


    def plot_state_lane(self, state_image_full, steps, fig, waypoints=[]):
        '''
        Plot lanes and way points
        '''
        # evaluate spline for 6 different spline parameters.
        t = np.linspace(0, 1, 6)
        lane_boundary1_points_points = np.array(splev(t, self.lane_boundary1_old))
        lane_boundary2_points_points = np.array(splev(t, self.lane_boundary2_old))
        
        plt.gcf().clear()
        plt.imshow(state_image_full[::-1])
        plt.plot(lane_boundary1_points_points[0], lane_boundary1_points_points[1]+120-self.cut_size, linewidth=5, color='orange')
        plt.plot(lane_boundary2_points_points[0], lane_boundary2_points_points[1]+120-self.cut_size, linewidth=5, color='orange')
        if len(waypoints):
            plt.scatter(waypoints[0], waypoints[1]+320-self.cut_size, color='white')

        plt.axis('off')
        plt.xlim((-0.5,320)) # 95.5 old max value
        plt.ylim((-0.5,240))
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        fig.canvas.flush_events()
