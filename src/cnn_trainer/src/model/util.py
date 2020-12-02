import numpy as np
import random
import math
import cv2
import os

from scipy.ndimage.filters import uniform_filter
from string import ascii_lowercase as LC
from matplotlib import pyplot as plt

PATH = '/home/fizzer/ros_ws/src/cnn_trainer'
PLATE_DIR = os.path.join(PATH, 'media', 'plates')
TEST_PATH = os.path.join(PATH, 'media', 'test_set')
MODEL_PATH = os.path.join(PATH, 'src', 'model')

VALIDATION_SPLIT = 0.2


LOWER_RED = np.array([0,165,100])
UPPER_RED = np.array([0,255,255])


# Get all files in a path
def files_in_folder(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


# Generate a one hot vector for a given character
def one_hot_letter(c):
    vec = [0] * 26
    lowercase = c.lower()
    index = LC.index(lowercase)
    vec[index] = 1
    return vec


# Generate a one hot vector for a given number 
def one_hot_num(n):
    vec = [0] * 10
    vec[n] = 1
    return vec


# Return an alphanumeric character from a given index
def index_to_val(i):
    abc123 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    return abc123[i]


# mask red letter
def thresh(frame):
    _, threshed = cv2.threshold(frame, 100, 255,cv2.THRESH_BINARY_INV)
    res = threshed.reshape(150,105,1)
    return res

# mask sample
def thresh2(frame):
    _, threshed = cv2.threshold(frame, 60, 255,cv2.THRESH_BINARY_INV)
    res = threshed.reshape(150,105,1)
    return res


# Return either 2 partitions of a plate image and their associated one hot vectors
def process_plate(my_file, model_type):
    plate_path = os.path.join(PLATE_DIR, my_file)
    plate_img = cv2.imread(plate_path, cv2.IMREAD_GRAYSCALE)
    
    # crop into subsections (x dataset) shape is (300,105,0)
    
    if model_type == 0:
        img1 = thresh(plate_img[140:290,45:150])
        img2 = thresh(plate_img[140:290,145:250])
        imgs = [img1,img2]
    else:
        img3 = thresh(plate_img[140:290,347:452])
        img4 = thresh(plate_img[140:290,447:552])
        imgs = [img3,img4]

    vecs = []
    
    if model_type == 0:
        for i in range(2):
            vecs.append(one_hot_letter(my_file[6+i]))
    else:
        for i in range(2):
            vecs.append(one_hot_num(int(my_file[8+i])))

    # for i,c in enumerate(imgs):
    #     print(vecs[i])
    #     cv2.imshow('c',c)
    #     cv2.waitKey(0)

    return imgs, vecs 


# Training datasets associated with the processed plates and their one-hot vectors
def get_training_dataset(model_type):
    plates = files_in_folder(PLATE_DIR)

    X_images = []
    Y_labels = []

    for p in plates:
        imgs, vecs = process_plate(p, model_type)
        X_images.extend(imgs)
        Y_labels.extend(vecs)

    X_dataset = np.array(X_images)
    Y_dataset = np.array(Y_labels)

    # for c in X_dataset:
    #     plt.imshow(c)
    #     plt.show()

    return X_dataset, Y_dataset # Returning the raw partitions.


# Returns 4 partitions of the license plate in the homographic image
def process_test_plate(my_file, model_type):
    img_path = os.path.join(TEST_PATH, my_file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # (329,405,1)
    img_r = cv2.resize(img, (400,400)) # 588, 450, 3

    sz = (105,150)

    if model_type == 0:
        img1 = thresh2(cv2.resize(img_r[300:-60,90:145], sz))
        img2 = thresh2(cv2.resize(img_r[300:-60,140:190], sz))
        imgs = [img1,img2]
    else:
        img3 = thresh2(cv2.resize(img_r[300:-60,220:270], sz))
        img4 = thresh2(cv2.resize(img_r[300:-60,270:315], sz))
        imgs = [img3,img4]

    # for i in imgs: 
    #     cv2.imshow('i',i)
    #     cv2.waitKey(0)

    return imgs


# Gets the testing dataset
def get_test_dataset(model_type):
    plates = files_in_folder(TEST_PATH)

    X_images = []

    for p in plates:
        imgs = process_test_plate(p, model_type)
        X_images.extend(imgs)

    X_dataset = np.array(X_images)

    # for c in X_dataset:
    #     plt.imshow(c)
    #     plt.show()

    return X_dataset # Returning the raw partitions.


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
    VARIABILITY = 5
    deviation = VARIABILITY*random.random()
    noise = np.random.normal(0, deviation, img.shape)
    img += noise

    img = uniform_filter(img,size=(35,35,1))
    np.clip(img, 0., 255.)
    return img

if __name__ == '__main__':
    x= get_test_dataset(1)

    for item in x:
        cv2.imshow('i', item)
        cv2.waitKey(0)
