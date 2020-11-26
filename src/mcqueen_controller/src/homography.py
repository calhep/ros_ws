#! /usr/bin/env python

import numpy as np
import cv2
import rospy

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


class Homography():

    def __init__(self, template_path):
        self.image_template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)


    def detect_features(self, frame):
        #Features
        sift = cv2.xfeatures2d.SIFT_create()
        kp_image, desc_image = sift.detectAndCompute(self.image_template, None)

        # Feature matching
        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # trainimage
        kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None)
        matches = flann.knnMatch(desc_image, desc_grayframe, k=2)
        good_points = [m for m, n in matches if m.distance < 0.6*n.distance]

        if len(good_points) > 10:
            query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
            train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
            
            matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()

            # Perspective transform
            h, w = self.image_template.shape
            # pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            # dst = cv2.perspectiveTransform(pts, matrix)

            print(dst)
            # cv2.imshow('gyuh', cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3))
            cv2.imshow('gyuh', cv2.warpPerspective(frame, matrix, (w, h)))
            cv2.waitKey(3)

            return cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
            
        # if len(good_points) > 0:
        #     return cv2.drawMatches(img,kp_image,grayframe,kp_grayframe,good_points,frame)

        else:
            return frame