#!/usr/bin/env python

import cv2
import numpy as np
import os
import random
import string


from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from random import randint
from PIL import Image, ImageFont, ImageDraw


# Paths to different directories in media containing either training/validation set or test set.
PATH = '/home/fizzer/ros_ws/src/cnn_trainer'
PLATE_PATH = os.path.join(PATH, 'media','plates')
TEST_DATA_DIR = os.path.join(PATH, 'media','test_set')
AUG_SET = os.path.join(PATH, 'media','aug_set')

# Choose directory to write to:
my_path = AUG_SET

# Specify augmentation
SHEAR = False
NOISE = False


# guh: https://stackoverflow.com/questions/43382045/keras-realtime-augmentation-adding-noise-and-contrast
def add_noise(img):
    '''Add random noise to an image'''
    VARIABILITY = 50
    deviation = VARIABILITY*random.random()
    noise = np.random.normal(0, deviation, img.shape)
    img += noise
    np.clip(img, 0., 255.)
    return img


def augment_plate(my_plate):
    datagen = ImageDataGenerator(
        preprocessing_function=add_noise,
    )

    images = [my_plate]
    img_arr = image.img_to_array(my_plate)
    img_arr = img_arr.reshape((1,) + img_arr.shape)

    test = datagen.flow(img_arr,batch_size=1)[0][0]

    return Image.fromarray(test, 'RGB')

for i in range(0, 1):

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
    my_plate = np.array(blank_plate_pil)

    # Apply noise/shear here if specified
    my_plate = augment_plate(my_plate)

    # Display image
    plt.imshow(my_plate)
    plt.show()
    
    # Write to file
    cv2.imwrite(os.path.join(my_path, "plate_{}{}.png".format(plate_alpha, plate_num)), my_plate)

print(len([name for name in os.listdir(my_path) if os.path.isfile(os.path.join(my_path, name))]) )
