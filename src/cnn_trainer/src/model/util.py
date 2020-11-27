import numpy as np
import cv2
import os

from string import ascii_lowercase as LC
from matplotlib import pyplot as plt

PATH = '/home/fizzer/ros_ws/src/cnn_trainer'
PLATE_DIR = os.path.join(PATH, 'media', 'plates')
MODEL_PATH = os.path.join(PATH, 'src', 'model')

VALIDATION_SPLIT = 0.2


# Get all files in a path
def files_in_folder(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


# Generate a one hot vector for a given character
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


# Return an alphanumeric character from a given index
def index_to_val(i):
    abc123 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    return abc123[i]


# Return the 4 partitions of a plate image and their associated one hot vectors
def process_plate(my_file):
    plate_path = os.path.join(PLATE_DIR, my_file)
    plate_img = cv2.imread(plate_path)
    
    # crop into subsections (x dataset)
    img1 = plate_img[49:349,45:150]
    img2 = plate_img[49:349,145:250]
    img3 = plate_img[49:349,347:452]
    img4 = plate_img[49:349,447:552]
    imgs = [img1,img2,img3,img4]

    vecs = []
    for i in range(4):
        vecs.append(one_hot(my_file[6+i]))

    return imgs, vecs


# Return the datasets associated with the processed plates and their one-hot vectors
def get_dataset():
    plates = files_in_folder(PLATE_DIR)

    X_images = []
    Y_labels = []

    for p in plates:
        imgs, vecs = process_plate(p)
        X_images.extend(imgs)
        Y_labels.extend(vecs)

    X_dataset = np.array(X_images) / 255  # normalize the data
    Y_dataset = np.array(Y_labels)

    return X_dataset, Y_dataset


# Prints info about datasets
def print_dataset_info(X_dataset, Y_dataset, vs):
    print("Total examples: {}\nTraining examples: {}\nValidation examples: {}".
            format(X_dataset.shape[0], 
                   math.ceil(X_dataset.shape[0] * (1-vs)),
                   math.floor(X_dataset.shape[0] * vs)))
    print("X shape: " + str(X_dataset.shape))
    print("Y shape: " + str(Y_dataset.shape))
