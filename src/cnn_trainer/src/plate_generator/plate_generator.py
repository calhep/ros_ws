#!/usr/bin/env python

import cv2
import numpy as np
import os
import random
import string

from random import randint
from PIL import Image, ImageFont, ImageDraw

PATH = '/home/fizzer/ros_ws/src/cnn_trainer'
PLATE_PATH = os.path.join(PATH, 'media','plates')

for i in range(0, 3):

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

    # Get a drawing context
    draw = ImageDraw.Draw(blank_plate_pil)
    monospace = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf", 200)
    draw.text((48, 105),plate_alpha + " " + plate_num, (255,0,0), font=monospace)

    # Convert back to OpenCV image and save
    blank_plate = np.array(blank_plate_pil)
    
    # Write to file
    cv2.imwrite(os.path.join(PLATE_PATH, "plate_{}{}.png".format(plate_alpha, plate_num)), blank_plate)

print(len([name for name in os.listdir(PLATE_PATH) if os.path.isfile(os.path.join(PLATE_PATH, name))]) )
