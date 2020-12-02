import numpy as np
import tensorflow as tf
import os
import cv2
import random
import math

from matplotlib import pyplot as plt
from scipy.ndimage.filters import uniform_filter
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

import util as util
import model as m

PATH = '/home/fizzer/ros_ws/src/cnn_trainer'
CAR_PATH = os.path.join(PATH, 'media','parking')
TEST_PATH = os.path.join(PATH, 'media', 'test_set')


# one hot vector
def car_one_hot(num):
    vecs = [0] * 8
    vecs[num-1] = 1
    return vecs


# process test pic
def process_test_pic(my_file):
    pic_path = os.path.join('/home/fizzer/ros_ws/src/mcqueen_controller/src/Homography_Matches/Sliced_Numbers/', my_file)
    img = cv2.imread(pic_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (100,130))
    # res = img_resized[200:390,180:-50]
    print(pic_path)
    print(img.shape)
    # plt.imshow(res)
    # plt.show()
    # res = res.reshape(190,150,1)
    img = img.reshape(img.shape[0], img.shape[1], 1)
    print(img.shape)
    return img


# process car pic
def process_car_pic(my_file):
    pic_path = os.path.join(CAR_PATH, my_file)
    img = cv2.imread(pic_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img,(200,600))
    img_resized = img_resized[235:365,100:] # 130, 100
    # plt.imshow(img)
    # plt.show()
    res = img_resized.reshape(130,100,1)

    return res


# gets and crops car pics
def get_car_datasets():
    files = util.files_in_folder(CAR_PATH)

    pics = []
    vecs = []

    # resize and crop to p_
    for f in files:
       # print("processing ", f)
        processed_pic = process_car_pic(f)
        pics.append(processed_pic)
        vecs.append(car_one_hot(int(f[0])))
        

    return pics, vecs


# Save a Keras model
def save_car_model(model):
    model.save('keras/car_model')


# Load a Keras model
def load_car_model():
    return models.load_model('keras/car_model')


# generate model for car
def generate_car_model(lr):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(130, 100, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(8, activation='softmax'))

    # Set Learning Rate and Compile Model
    model.compile(loss='categorical_crossentropy',
                    optimizer=optimizers.RMSprop(lr=lr),
                    metrics=['acc'])

    return model


# Get car model either by generating a new one or load from local
def get_car_model(lr=1e-4, new=False):  

    if new:
        print("compiling new model")
        model = generate_car_model(lr)
    else:
        print("Loading local model")
        model = load_car_model()

    model.summary()

    return model


def add_noise(img):
    # VARIABILITY = 1
    # deviation = VARIABILITY*random.random()
    # noise = np.random.normal(0, deviation, img.shape)
    # img += noise

    blur_factor = random.randint(0,5)

    img = uniform_filter(img,size=(blur_factor,blur_factor,1))
    np.clip(img, 0., 255.)
    return img


def visualize_idg(aug, X_dataset):
    for data in X_dataset:
        sample = np.expand_dims(data,0)
        it = aug.flow(sample, batch_size=1)
        
        # generate batch
        batch = it.next()

        # convert to uint8
        image = batch[0].astype('uint8')
        cv2.imshow('a',image)
        cv2.waitKey(0)


# Predict car
def train_car_model(model, X_dataset, Y_dataset, vs, epochs):
    # print(X_dataset.shape)
    # print(Y_dataset.shape)
    
    aug = ImageDataGenerator(
        shear_range=2,
        # rotation_range=5,
        zoom_range=[1,3.5],
        width_shift_range=[-30,30],
        height_shift_range=[-20,20],
        preprocessing_function=add_noise,
        brightness_range=(0.3,1.3),
        validation_split=vs
    )

    print("Visualizing IDG.")
    #visualize_idg(aug, X_dataset) # VIS

    training_dataset = aug.flow(X_dataset, Y_dataset, subset='training')
    validation_dataset = aug.flow(X_dataset, Y_dataset, subset='validation')

    history_conv = model.fit(
        training_dataset,
        steps_per_epoch=105,
        batch_size=4,
        epochs=epochs,
        verbose=1,
        validation_data=validation_dataset
    )

    print('Model trained.')

    # Fit the data and get history of the model over time, if specified
    plt.plot(history_conv.history['loss'])
    plt.plot(history_conv.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training loss', 'Validation loss'], loc='upper right')
    plt.show()

    plt.plot(history_conv.history['acc'])
    plt.plot(history_conv.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epoch')
    plt.legend(['Train accuracy', 'Val accuracy'], loc='lower right')
    plt.show()

    return model


# predict the car
def predict_car(model, car):
    # pic = process_car_pic(file)
    image = np.expand_dims(car,axis=0)

    predicted_car = model.predict(image)[0]
    index_pred = np.argmax(predicted_car)

    res = [0] * 8
    res[index_pred] = 1
    print("Predicted: ", index_pred+1)
    print("Confidence: ", predicted_car)
    print("\n")


def main():
    NEW_MODEL = True
    TRAIN = True

    EPOCHS = 30
    VS = 0.2

    imgs, vecs = get_car_datasets()
    X_dataset = np.array(imgs)
    Y_dataset = np.array(vecs)
    print(len(X_dataset))

    # for x in X_dataset:
    #     cv2.imshow('x',x)
    #     cv2.waitKey(0)

    print("Total examples: {}\nTraining examples: {}\nTest examples: {}".
      format(X_dataset.shape[0],
             math.ceil(X_dataset.shape[0] * (1-VS)),
             math.floor(X_dataset.shape[0] * VS)))
    print("X shape: " + str(X_dataset.shape))
    print("Y shape: " + str(Y_dataset.shape))

    model = get_car_model(lr=1e-4,new=NEW_MODEL)

    if TRAIN:
        model = train_car_model(model,
            X_dataset,
            Y_dataset,
            VS,
            EPOCHS,
        )

    save_car_model(model)

    # predict car from validation set xd bad practice
    print("predicting from vali")
    files = util.files_in_folder(CAR_PATH)
    file_to_test = files[6]

    car = process_car_pic(file_to_test)
    print("actual: ", file_to_test)
    predict_car(model, car)

    # predict car from test_set O_O
    print("now predicting from test set")
    tests = util.files_in_folder('/home/fizzer/ros_ws/src/mcqueen_controller/src/Homography_Matches/Sliced_Numbers/')
    for t in tests:
        my_test = process_test_pic(t)
        print('actual: ', t[7:8])
        cv2.imshow('g', my_test)
        cv2.waitKey(3)
        predict_car(model, my_test)


if __name__ == '__main__':
    main()
