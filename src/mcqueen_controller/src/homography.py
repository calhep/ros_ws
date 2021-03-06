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

    def __init__(self, pr, model_l, model_n):
        self.pr = pr
        self.model = models.load_model('/home/fizzer/ros_ws/src/cnn_trainer/src/model/keras/car_model')
        self.model_l = model_l
        self.model_n = model_n

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
        self.flag = False
    
    
    # Callback function for homography subscriber
    def callback(self, image):
        car_homography = self.run_homography(image, True)

        if self.flag:
            self.pr.stop_comp()
        
        elif car_homography is not None:
            # plate_homography = self.run_homography(car_homography, False)
            # self.splice_homography(car_homography)
            processed_number = self.slice_number(car_homography)
            processed_plate = self.slice_plate(car_homography)

            max_pred, pred_number = self.generate_prediction(processed_number)

            if max_pred > 0.97:
                var = self.plate_num

                if pred_number == 4 or pred_number == 5:
                    dilate = True
                    erode = False
                elif pred_number == 6:
                    erode = True
                    dilate = False
                else:
                    erode = False
                    dilate = False
                
                processed_plate = processed_plate.reshape(processed_plate.shape[0],processed_plate.shape[1],1)
                predicted_plate = self.plate_prediction(processed_plate, dilate,erode)

                if not self.flag:
                    self.plate_num = self.plate_reference[str(pred_number)]

                if var == 5 and not self.plate_num == 5:
                    self.plate_num = 6
                    pred_number = 1
                    self.flag = True
                

                print(self.plate_num)
                
                self.pr.publish_plate(pred_number, predicted_plate) # TODO: put predicted plate



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
        return plate_slice
    
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


    # prediction for plate
    def plate_prediction(self, image,dilate=False,erode=False):
        letters, nums = self.process_plate(image,dilate,erode)

        # predict letters
        l1 = np.expand_dims(letters[0],axis=0)
        l2 = np.expand_dims(letters[1],axis=0)

        with graph.as_default():
            backend.set_session(sess)
            pred_l1 = self.model_l.predict(l1)[0]
            pred_l2 = self.model_l.predict(l2)[0]

        i1 = np.argmax(pred_l1)
        i2 = np.argmax(pred_l2)

        print("letter confidence")
        print(pred_l1)
        print(pred_l2)

        char1 = self.index_to_val(i1)
        char2 = self.index_to_val(i2)

        pred_chars = char1 + char2

        # predict numbers
        n1 = np.expand_dims(nums[0],axis=0)
        n2 = np.expand_dims(nums[1],axis=0)

        with graph.as_default():
            backend.set_session(sess)
            pred_n1 = self.model_n.predict(n1)[0]
            pred_n2 = self.model_n.predict(n2)[0]

        i3 = np.argmax(pred_n1)
        i4 = np.argmax(pred_n2)

        print("number confidence")
        print(pred_n1)
        print(pred_n2)

        pred_chars = pred_chars + str(i3) + str(i4)

        print(pred_chars)
        return pred_chars



    def process_img(self, image):
        img = cv2.resize(image, (100,130))
        _, img = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY_INV)
        img = img.reshape(img.shape[0], img.shape[1], 1)

        return img

    def thresh_char(self, frame,dilate,erode,more=0):
        _, threshed = cv2.threshold(frame, 60, 255,cv2.THRESH_BINARY_INV)

        if dilate:
            kernel = np.ones((5+more,5+more),np.uint8)
            threshed = cv2.dilate(threshed,kernel,iterations=1)

        if erode:
            kernel = np.ones((4+more,4+more),np.uint8)
            threshed = cv2.erode(threshed,kernel,iterations=1)

        res = threshed.reshape(150,105,1)
        return res
  

    def process_plate(self, image,dilate,erode):
        print(image.shape)

        sz = (105,150)
        
        letter1 = self.thresh_char(cv2.resize(image[10:-10,10:50],sz), dilate, erode,more=0)
        letter2 = self.thresh_char(cv2.resize(image[10:-10,50:100],sz), dilate,erode,more=0)
    
        num1 = self.thresh_char(cv2.resize(image[10:-10,130:175],sz),dilate,erode,more=0)
        num2 = self.thresh_char(cv2.resize(image[10:-10,178:213],sz),dilate,erode, more=2)

        cv2.imshow('a',letter1)
        cv2.imshow('b',letter2)
        cv2.imshow('c',num1)
        cv2.imshow('d',num2)

        letters = [letter1, letter2]
        nums = [num1, num2]

        return letters, nums

    
    # Return an alphanumeric character from a given index
    def index_to_val(self, i):
        abc123 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        return abc123[i]
