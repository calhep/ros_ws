#! /usr/bin/env python

import numpy as np
import cv2
import rospy

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


class Homography():

    # TODO: make sure this works lmfao

    def __init__(self):
        self.image_sub = rospy.Subscriber('/R1/pi_camera/image_raw',Image,self.callback)
        self.bridge = CvBridge()


    # Source: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
    def callback(self, image):
        try:
            frame = self.bridge.imgmsg_to_cv2(image, 'mono8') # image grayscaled
        except CvBridgeError as e:
            print(e)

        cam_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

        img = cv2.imread(self.template_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # grayscaled img

        # Instantiate SIFT object
        sift = cv2.xfeatures2d.SIFT_create()

        # find keypoints and descriptors
        kp_img, des_img = sift.detectAndCompute(img_gray, None)
        kp_cam, des_cam = sift.detectAndCompute(cam_gray, None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        MIN_MATCH_COUNT = 10
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        # FLANN 
        matches = cv2.FlannBasedMatcher(index_params, search_params).knnMatch(des_img, des_cam, k=2)
        matches_mask = [[0,0] for i in xrange(len(matches))]

        good = []
        # apply Lowe's ratio test
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matches_mask[i] = [1,0]
                good.append(m)
        
        draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matches_mask,
                   flags = 0)

        # enough valid keypoints to form a homography
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([ kp_img[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp_cam[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            h, w = img_gray.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts, matrix)

            out = cv2.polylines(frame, [np.int32(dst)], True, 255,3, cv2.LINE_AA)

        # else, map out any keypoints, since we can't form a homography
        else:
            out = cv2.drawMatchesKnn(img, kp_img, frame, kp_cam, matches, None, **draw_params)

        # display image on webcam
        pixmap = self.convert_cv_to_pixmap(out)
        self.live_image_label.setPixmap(pixmap)