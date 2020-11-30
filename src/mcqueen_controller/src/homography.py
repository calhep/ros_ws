#! /usr/bin/env python

import numpy as np
import cv2

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


MIN_MATCHES = 20

class Homography():

    def __init__(self, template_paths):
        self.image_templates = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in template_paths]
        

    def detect_features(self, grayframe, plate_num):
        # grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # trainimage
        sift = cv2.xfeatures2d.SIFT_create()

        # generate keypoints and descriptors
        kp_image, desc_image = sift.detectAndCompute(self.image_templates[plate_num], None)
        kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None)

        # Feature matching
        index_params = dict(algorithm=0, trees=5)
        search_params = dict()

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc_image, desc_grayframe, k=2)

        good_points = [m for m, n in matches if m.distance < 0.6*n.distance]

        if len(good_points) >= MIN_MATCHES:
            query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
            train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
            
            matrix, mask = cv2.findHomography(train_pts, query_pts, cv2.RANSAC, 5.0)

            # Perspective transform
            h, w = self.image_templates[plate_num].shape
            # pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            # dst = cv2.perspectiveTransform(pts, matrix)

            cv2.imshow('gyuh', cv2.warpPerspective(grayframe, matrix, (w, h)))
            cv2.waitKey(3)
            print(len(good_points))
            # return cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)

            return True
            
        # if len(good_points) > 0:
        #     return cv2.drawMatches(img,kp_image,grayframe,kp_grayframe,good_points,frame)

        else:
            print("Too few valid keypoints found: {}/{}".format(len(good_points), MIN_MATCHES))

            return False