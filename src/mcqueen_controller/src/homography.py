#! /usr/bin/env python

import rospy
import sys
import cv2
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


MIN_CAR_MATCHES = 14
MIN_PLATE_MATCHES = 8


class Homography():

    def __init__(self):
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.image_sub = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.callback, queue_size=1, buff_size=1000000)
        self.bridge = CvBridge()

        # Car image references
        self.template_car_paths = ['/home/fizzer/ros_ws/src/mcqueen_controller/src/Homography_Templates/Car_Templates/p{}.png'.format(x + 1) for x in range(6)]
        self.car_templates = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in self.template_car_paths]
        self.kp_desc_images = [(lambda x: self.sift.detectAndCompute(x, None))(x) for x in self.car_templates]

        # Plate image references
        self.template_plate_paths = ['/home/fizzer/ros_ws/src/mcqueen_controller/src/Homography_Templates/Plate_Templates/p{}.png'.format(x + 1) for x in range(6)]
        self.plate_templates = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in self.template_plate_paths]
        self.kp_desc_plates = [(lambda x: self.sift.detectAndCompute(x, None))(x) for x in self.plate_templates]

        # Initialize sift parameters
        self.index_params = {'algorithm':0, 'trees':5}
        self.search_params = {}
        self.flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)

        self.plate_num = 0
    
    
    # Callback function for homography subscriber
    def callback(self, image):
        car_homography = self.run_homography(image, True)

        if car_homography is not None:
            # plate_homography = self.run_homography(car_homography, False)
            # self.splice_homography(car_homography)
            self.slice_number(car_homography)

    
    # Method for generating homography on an image
    def run_homography(self, grayframe, detecting_car):

        # Determine mode for homography
        if detecting_car:
            reference_image = self.car_templates[self.plate_num]
            kp_image, desc_image = self.kp_desc_images[self.plate_num]

            grayframe = self.bridge.imgmsg_to_cv2(grayframe, 'mono8')
            grayframe = grayframe[300:-100,:350] # Originally 1280 x 720
            w,h = grayframe.shape 
            grayframe = cv2.resize(grayframe,(int(1*h),int(1*w))) # 320 x 180

            min_matches = MIN_CAR_MATCHES

            cv2.imshow('reference_image', reference_image)
            cv2.imshow('grayframe', grayframe)
            cv2.waitKey(3)

        else:
            reference_image = self.plate_templates[self.plate_num]
            kp_image, desc_image = self.kp_desc_plates[self.plate_num]

            min_matches = MIN_PLATE_MATCHES

        # generate keypoints and descriptors
        kp_grayframe, desc_grayframe = self.sift.detectAndCompute(grayframe, None)

        # Feature matching
        matches = self.flann.knnMatch(desc_image, desc_grayframe, k=2)
        good_points = [m for m, n in matches if m.distance < 0.6*n.distance]

        if len(good_points) >= min_matches:
            query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
            train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
            
            matrix, mask = cv2.findHomography(train_pts, query_pts, cv2.RANSAC, 5.0)

            # Perspective transform
            h, w = reference_image.shape
            hom_match = cv2.warpPerspective(grayframe, matrix, (w, h))

            if detecting_car:
                cv2.imwrite('/home/fizzer/ros_ws/src/mcqueen_controller/src/Homography_Matches/match_p{}.jpg'.format(self.plate_num), hom_match)
                cv2.imshow('car_homography', hom_match)

            else:
                cv2.imshow('plate_homography', hom_match)

            cv2.waitKey(3)
            print(len(good_points))
            self.plate_num += 1
            
            return hom_match

        else:
            print("Too few valid keypoints found: {}/{}".format(len(good_points), min_matches))

            return None


    # Method for splitting up homographic match
    def slice_plate(self, image):
        h, w = image.shape

        plate_slice = image[int(0.75*h):-1*int(0.05*h),int(0.2*w):-1*int(0.2*w)]
        cv2.imshow('gyuh', plate_slice)
        return
    
    # Method for creating number slice
    def slice_number(self, image):
        h, w = image.shape

        number_slice = image[int(0.3*h):-1*int(0.3*h),int(0.2*w):-1*int(0.2*w)]
        cv2.imshow('gyuh', number_slice)
        return

    # Image contouring
    def image_contour(self, image):
        ret, thresh = cv2.threshold(img,127,255,0)
        contours, hierarchy = cv2.findContours(thresh, 1, 2)

        return

    # Method for getting the prediction of a homographic match
    def generate_prediction(self, image):
        return