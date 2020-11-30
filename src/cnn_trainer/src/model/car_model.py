import numpy as np
import tensorflow as tf
import os
import cv2
from matplotlib import pyplot as plt

from matplotlib import pyplot as plt
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

import util as util
import model as m

PATH = '/home/fizzer/ros_ws/src/cnn_trainer'
CAR_PATH = os.path.join(PATH, 'media','test_set')


# one hot vector
def car_one_hot(num):
    vecs = [0] * 8
    vecs[num-1] = 1
    return vecs


def process_car_pic(my_file):
    pic_path = os.path.join(CAR_PATH, my_file)
    img = cv2.imread(pic_path)
    resized_img = cv2.resize(img,(100,150)) # w,h
    return resized_img[60:100]


# gets and crops car pics
def get_car_datasets():
    files = util.files_in_folder(CAR_PATH)

    pics = []
    vecs = []

    # resize and crop to p_
    for f in files:
        processed_pic = process_car_pic(f)
        pics.append(processed_pic)
        print(f[0])
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
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(40, 100, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(36, activation='softmax'))

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


# TODO: Implement idg and train network
def train_car_model(model, X_dataset, Y_dataset, vs, epochs):
    # print(X_dataset.shape)
    # print(Y_dataset.shape)
    
    aug = ImageDataGenerator(
        shear_range=0.3,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=[-15,15],
        preprocessing_function=util.add_noise,
        brightness_range=[0.25,1.3],
        validation_split=vs
    )

    print("Visualizing IDG.")
    #m.visualize_idg(aug, X_dataset)

    training_dataset = aug.flow(X_dataset, Y_dataset, subset='training')
    validation_dataset = aug.flow(X_dataset, Y_dataset, subset='validation')

    history_conv = model.fit(
        training_dataset,
        batch_size=32,
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


# Visualize the output from the ImageDataGenerator. This can probably be 
def visualize_idg(aug, X_dataset):
    for data in X_dataset:
        sample = np.expand_dims(data,0)
        it = aug.flow(sample, batch_size=1)
        
        # generate batch
        batch = it.next()

        # convert to uint8
        image = batch[0].astype('uint8')
        plt.imshow(image)
        plt.show()


# predict the car
def predict_car(model, my_file):
    print("actual: ", my_file)
    pic = process_car_pic(file)
    image = np.expand_dims(pic,axis=0)
    predicted_car = model.predict(image)[0]
    print("predicted: ", predicted_car)


def main():
    NEW_MODEL = True

    imgs, vecs = get_car_datasets()
    X_dataset = np.array(imgs)
    Y_dataset = np.array(vecs)

    model = get_car_model(lr=1e-4,new=NEW_MODEL)

    model = train_car_model(model,
        X_dataset,
        Y_dataset,
        0.2,
        10,
    )

    save_car_model(model)

    # try predict lol
    predict_car()

if __name__ == '__main__':
    main()

