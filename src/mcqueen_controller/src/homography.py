#! /usr/bin/env python

import rospy
import sys
import cv2
import numpy as np

from tensorflow.compat.v1 import get_default_graph
from tensorflow.python.keras import models
from tensorflow.python.keras import backend
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


MIN_CAR_MATCHES = 14
MIN_PLATE_MATCHES = 8

sess = backend.get_session()
graph = get_default_graph()


class Homography():

    def __init__(self, pr):
        self.pr = pr
        self.model = models.load_model('/home/fizzer/ros_ws/src/cnn_trainer/src/model/keras/car_model')

        self.sift = cv2.xfeatures2d.SIFT_create()
        self.image_sub = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.callback, queue_size=1, buff_size=1000000)
        self.bridge = CvBridge()

        # Car image references
        self.template_car_paths = ['/home/fizzer/ros_ws/src/mcqueen_controller/src/Homography_Templates/Car_Templates/p{}.png'.format(x + 1) for x in range(8)]
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

        self.plate_reference = {'2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '1': 6, '7': 7, '8': 7}
        self.plate_num = 0
    
    
    # Callback function for homography subscriber
    def callback(self, image):
        car_homography = self.run_homography(image, True)

        if car_homography is not None:
            # plate_homography = self.run_homography(car_homography, False)
            # self.splice_homography(car_homography)
            processed_number = self.slice_number(car_homography)
            procssed_plate = self.slice_plate(car_homography)

            image_contours = self.get_image_contour(processed_plate)

            max_pred, pred_number = self.generate_prediction(processed_number)

            if max_pred > 0.95:
                self.plate_num = self.plate_reference[str(pred_number)]
                print(self.plate_num)

                self.pr.publish_plate(pred_number, 'ABCD')


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
                # cv2.imshow('car_homography', hom_match)

            # else:
            #     cv2.imshow('plate_homography', hom_match)

            # cv2.waitKey(3)
            print(len(good_points))
            
            return hom_match

        else:
            print("Too few valid keypoints found: {}/{}".format(len(good_points), min_matches))

            return None


    # Method for splitting up homographic match
    def slice_plate(self, image):
        h, w = image.shape

        plate_slice = image[int(0.75*h):-1*int(0.05*h),int(0.2*w):-1*int(0.2*w)]
        # cv2.imshow('gyuh', plate_slice)
        return
    
    # Method for creating number slice
    def slice_number(self, image):
        h, w = image.shape

        number_slice = image[int(0.25*h):-1*int(0.25*h),int(0.5*w):-1*int(0.2*w)]
        cv2.imshow('gyuh', number_slice)
        cv2.imwrite('/home/fizzer/ros_ws/src/mcqueen_controller/src/Homography_Matches/Sliced_Numbers/slice_p{}.jpg'.format(self.plate_num), number_slice)
        
        return number_slice

    # Method for getting the prediction of a homographic match
    def generate_prediction(self, image):
        img = self.process_img(image)
        img = np.expand_dims(img, axis=0)

        with graph.as_default():
            backend.set_session(sess)
            predicted_car = self.model.predict(img)[0]

        index_pred = np.argmax(predicted_car)

        res = [0] * 6
        res[index_pred] = 1

        print("Predicted: ", index_pred + 1)
        print("Confidence: ", predicted_car)

        res_max = np.amax(predicted_car)

        return (res_max, index_pred + 1)


    def process_img(self, image):
        img = cv2.resize(image, (100,130))
        _, img = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY_INV)
        img = img.reshape(img.shape[0], img.shape[1], 1)

        return img


    # Image contouring
    def get_image_contour(self, image):
        processed_plate = self.plate_processing(image)
        cropped_characters = self.plate_contouring(processed_plate)

        return


    # Code inspired by https://medium.com/@quangnhatnguyenle/detect-and-recognize-vehicles-license-plate-with-machine-learning-and-python-part-2-plate-de644de9849f
    def plate_processing(self, image):
        # Apply thresholding 
        binary_thresh = cv2.threshold(image, 180, 255,
                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        ## Apply dilation 
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh_morph = cv2.morphologyEx(binary_thresh, cv2.MORPH_DILATE, morph_kernel)

        return thresh_morph


    # Code inspired by https://medium.com/@quangnhatnguyenle/detect-and-recognize-vehicles-license-plate-with-machine-learning-and-python-part-2-plate-de644de9849f
    def plate_contouring(self, image):
        cont, _  = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # creat a copy version "test_roi" of plat_image to draw bounding box
        test_roi = plate_image.copy()

        # Initialize a list which will be used to append charater image
        crop_characters = []

        # define standard width and height of character
        digit_w, digit_h = 30, 60

        for c in self.sort_contours(cont):
            (x, y, w, h) = cv2.boundingRect(c)
            ratio = h/w
            if 1<=ratio<=3.5: # Only select contour with defined ratio
                if h/plate_image.shape[0]>=0.5: # Select contour which has the height larger than 50% of the plate
                    # Draw bounding box arroung digit number
                    cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 2)

                    # Sperate number and gibe prediction
                    curr_num = thre_mor[y:y+h,x:x+w]
                    curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                    _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    crop_characters.append(curr_num)
        
        print("Detect {} letters...".format(len(crop_characters)))   
        return crop_characters


    # Code inspired by https://medium.com/@quangnhatnguyenle/detect-and-recognize-vehicles-license-plate-with-machine-learning-and-python-part-2-plate-de644de9849f    # Create sort_contours() function to grab the contour of each digit from left to right
    def sort_contours(cnts,reverse = False):
        i = 0

        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: b[1][i], reverse=reverse))
        
        return cnts
