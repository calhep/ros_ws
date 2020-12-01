#! /usr/bin/env python

import rospy
import sys
import cv2
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


MIN_MATCHES = 15

class Homography():

    def __init__(self):
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.image_sub = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.callback, queue_size=1, buff_size=1000000)
        self.bridge = CvBridge()

        self.template_paths = self.image_paths = ['/home/fizzer/ros_ws/src/mcqueen_controller/src/media/New/p{}.png'.format(x + 1) for x in range(6)]
        self.image_templates = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in self.template_paths]
        self.kp_desc_images = [(lambda x: self.sift.detectAndCompute(x, None))(x) for x in self.image_templates]

        self.index_params = {'algorithm':0, 'trees':5}
        self.search_params = {}
        self.flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)

        self.plate_num = 0
    
    def generate_keypoints(self, image):
        grayframe = self.bridge.imgmsg_to_cv2(image, 'mono8')
        grayframe = grayframe[300:-100,:250] # Originally 1280 x 720
        w,h = grayframe.shape 

        grayframe = cv2.resize(grayframe,(int(1*h),int(1*w))) # 320 x 180

        cv2.imshow('ga', self.image_templates[self.plate_num])
        cv2.imshow('uwu', grayframe)
        cv2.waitKey(3)

        # generate keypoints and descriptors
        kp_image, desc_image = self.kp_desc_images[self.plate_num]
        # kp_image, desc_image = self.kp_desc_image
        kp_grayframe, desc_grayframe = self.sift.detectAndCompute(grayframe, None)

        # Feature matching
        matches = self.flann.knnMatch(desc_image, desc_grayframe, k=2)
        good_points = [m for m, n in matches if m.distance < 0.6*n.distance]

        if len(good_points) >= MIN_MATCHES:
            query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
            train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
            
            matrix, mask = cv2.findHomography(train_pts, query_pts, cv2.RANSAC, 5.0)

            # Perspective transform
            h, w = self.image_templates[self.plate_num].shape
            # pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            # dst = cv2.perspectiveTransform(pts, matrix)

            cv2.imshow('gyuh', cv2.warpPerspective(grayframe, matrix, (w, h)))
            cv2.waitKey(3)
            print(len(good_points))
            # cv2.imshow('gyuh',cv2.polylines(grayframe, [np.int32(dst)], True, (255, 0, 0), 3))
            self.plate_num += 1
            
        # if len(good_points) > 0:
        #     return cv2.drawMatches(img,kp_image,grayframe,kp_grayframe,good_points,frame)

        else:
            print("Too few valid keypoints found: {}/{}".format(len(good_points), MIN_MATCHES))

    def callback(self, image):
        self.generate_keypoints(image)