import math
import numpy as np
import re
import cv2
import os

from string import ascii_lowercase as LC
from matplotlib import pyplot as plt

PATH = '/home/fizzer/ros_ws/src/cnn_trainer'
PLATE_DIR = os.path.join(PATH, 'media', 'plates')


def files_in_folder(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def one_hot(c):
    vec = [0] * 36
    try:
        int(c)
        index = int(c) + 26
    except ValueError as e:
        lowercase = c.lower()
        index = LC.index(lowercase)
    vec[index] = 1
    return vec


def process(my_file):
    plate_path = os.path.join(PLATE_DIR, my_file)
    plate_img = cv2.imread(plate_path)
    # plt.imshow(plate_img)
    # plt.show()
    # crop into subsections (x dataset)
    print(plate_img.shape)



def main():
    plates = files_in_folder(PLATE_DIR)
    process(plates[0])



if __name__ == '__main__':
    main()
