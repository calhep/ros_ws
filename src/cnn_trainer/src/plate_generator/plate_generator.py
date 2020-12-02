#!/usr/bin/env python

import cv2
import numpy as np
import os
import random
import string

from matplotlib import pyplot as plt
from random import randint
from PIL import Image, ImageFont, ImageDraw


# Paths to different directories containing either training/validation set or test set.
PATH = '/home/fizzer/ros_ws/src/cnn_trainer'
PLATE_PATH = os.path.join(PATH, 'media','plates')
TEST_DATA_DIR = os.path.join(PATH, 'media','test_set')
PARKING_PATH = os.path.join(PATH,'media','parking')

# Choose directory to write to:
my_path = PLATE_PATH


for i in range(0, 15):

    # number to write to parking identifier
    #id = randint(1,8)
    id = 6

    # Pick two random letters
    plate_alpha = ""
    for _ in range(0, 2):
        plate_alpha += (random.choice(string.ascii_uppercase))

    # Pick two random numbers
    num = randint(0, 99)
    plate_num = "{:02d}".format(num)

    # Write plate to image
    blank_plate_path = os.path.join(PATH, 'media','blank_plate.png')
    blank_plate = cv2.imread(blank_plate_path)

    # Convert into a PIL image (this is so we can use the monospaced fonts)
    blank_plate_pil = Image.fromarray(blank_plate)



    # Create parking spot label
    s = "P" + str(id)
    print(s)
    parking_spot = 255 * np.ones(shape=[600, 600, 3], dtype=np.uint8)
    cv2.putText(parking_spot, s, (30, 450), cv2.FONT_HERSHEY_PLAIN, 28,
                    (0, 0, 0), 30, cv2.LINE_AA)
    

    # Get a drawing context
    draw = ImageDraw.Draw(blank_plate_pil)
    monospace = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf", 200)
    draw.text((48, 105),plate_alpha + " " + plate_num, (255,0,0), font=monospace)

    # Convert back to OpenCV image and save
    my_plate = np.array(blank_plate_pil)

    # Display image
    # plt.imshow(my_plate)
    # plt.show()

    # Concatenate the parking number and plate together
    spot_w_plate = np.concatenate((parking_spot, my_plate), axis=0)

    # unlabelled plates (no QR)
    unlabelled = np.concatenate((255 * np.ones(shape=[600, 600, 3],
                                    dtype=np.uint8), spot_w_plate), axis=0)
                                    
    # Write to file
    if my_path == PARKING_PATH:
        cv2.imwrite(os.path.join(my_path, "{}plate_{}{}.png".format(id, plate_alpha, plate_num)), unlabelled)
    elif my_path == PLATE_PATH:
        cv2.imwrite(os.path.join(my_path, "plate_{}{}.png".format(plate_alpha, plate_num)), my_plate)

print(len([name for name in os.listdir(my_path) if os.path.isfile(os.path.join(my_path, name))]) )
