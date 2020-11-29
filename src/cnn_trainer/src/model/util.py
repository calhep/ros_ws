import numpy as np
import random
import math
import cv2
import os

from string import ascii_lowercase as LC
from matplotlib import pyplot as plt

PATH = '/home/fizzer/ros_ws/src/cnn_trainer'
PLATE_DIR = os.path.join(PATH, 'media', 'plates')
TEST_PATH = os.path.join(PATH, 'media', 'test_set')
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
    
    # crop into subsections (x dataset) shape is (300,105,0)
    img1 = plate_img[49:349,45:150]
    img2 = plate_img[49:349,145:250]
    img3 = plate_img[49:349,347:452]
    img4 = plate_img[49:349,447:552]
    imgs = [img1,img2,img3,img4]

    vecs = []
    for i in range(4):
        vecs.append(one_hot(my_file[6+i]))

    for c in imgs:
        plt.imshow(c)
        plt.show()

    return imgs, vecs


# Training datasets associated with the processed plates and their one-hot vectors
def get_training_dataset():
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


# Returns 4 partitions of the license plate in the homographic image
def process_homographic_plate(my_file):
    img_path = os.path.join(TEST_PATH, my_file)
    img = cv2.imread(img_path) # (150,196,3)
    h, w, _ = img.shape
    img_upscaled = cv2.resize(img, (3*h,3*w)) # 588, 450, 3

    char1 = cv2.resize(img_upscaled[430:535,75:130], (105,300))
    char2 = cv2.resize(img_upscaled[430:535,120:180], (105,300))
    char3 = cv2.resize(img_upscaled[430:535,216:269], (105,300))
    char4 = cv2.resize(img_upscaled[430:535,260:317], (105,300))

    chars = [char1, char2, char3, char4]

    # for c in chars:
    #     plt.imshow(c)
    #     plt.show()
    print(char1.shape)
    return chars


# Gets the testing dataset
def get_test_dataset():
    # TODO: make this a workflow
    plates = files_in_folder(TEST_PATH)

    test = []


# Prints info about datasets
def print_dataset_info(X_dataset, Y_dataset, vs):
    print("Total examples: {}\nTraining examples: {}\nValidation examples: {}".
            format(X_dataset.shape[0], 
                   math.ceil(X_dataset.shape[0] * (1-vs)),
                   math.floor(X_dataset.shape[0] * vs)))
    print("X shape: " + str(X_dataset.shape))
    print("Y shape: " + str(Y_dataset.shape))


# Generate gaussian noise in img
# https://stackoverflow.com/questions/43382045/keras-realtime-augmentation-adding-noise-and-contrast
def add_noise(img):
    VARIABILITY = 0.9
    deviation = VARIABILITY*random.random()
    noise = np.random.normal(0, deviation, img.shape)
    img += noise
    np.clip(img, 0., 255.)
    return img

if __name__ == '__main__':
    get_test_dataset()
