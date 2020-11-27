import math
import numpy as np
import os

from matplotlib import pyplot as plt
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import plot_model

import util as util


PATH = '/home/fizzer/ros_ws/src/cnn_trainer'
PLATE_DIR = os.path.join(PATH, 'media', 'plates')
MODEL_PATH = os.path.join(PATH, 'src', 'model')


# Save a Keras model
def save_model(model):
    model.save('keras/my_model')
    return


# Load a Keras model
def load_model():
    return models.load_model('keras/my_model')


# Generate completely new training model
def generate_model(lr):
    conv_model = models.Sequential()
    conv_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 105, 3)))
    conv_model.add(layers.MaxPooling2D((2, 2)))
    conv_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    conv_model.add(layers.MaxPooling2D((2, 2)))
    conv_model.add(layers.Flatten())
    conv_model.add(layers.Dropout(0.5))
    conv_model.add(layers.Dense(512, activation='relu'))
    conv_model.add(layers.Dense(36, activation='softmax'))

    # Set Learning Rate and Compile Model
    conv_model.compile(loss='categorical_crossentropy',
                    optimizer=optimizers.RMSprop(lr=lr),
                    metrics=['acc'])

    return conv_model


# Get a model either by generating a new one or load from local
def get_model(X_dataset, Y_dataset, lr=1e-4, vs=0.2, print_summary=False, new=False):  
    
    print("Total examples: {}\nTraining examples: {}\nTest examples: {}".
            format(X_dataset.shape[0], 
                   math.ceil(X_dataset.shape[0] * (1-vs)),
                   math.floor(X_dataset.shape[0] * vs)))
    print("X shape: " + str(X_dataset.shape))
    print("Y shape: " + str(Y_dataset.shape))

    # Create a new model and save it or load one locally.
    if new:
        print("compiling new model")
        conv_model = generate_model(lr)
        save_model(conv_model)
    else:
        print("Loading local model")
        conv_model = load_model()

    conv_model.summary()

    return conv_model


# Predict a plate using a model
def predict_plate(plate, model):
    imgs, vecs = util.process_plate(plate)
    dataset = np.array(imgs) / 255

    chars = []

    for i in range(4):
        image = np.expand_dims(dataset[i], axis=0)

        y_true = vecs[i]
        index_true = np.argmax(y_true)

        y_predicted = model.predict(image)[0]
        index_predicted = np.argmax(y_predicted)

        chars.append(util.index_to_val(index_predicted))

    print("Predicted:", chars) 


def main():
    # PARAMETERS TO ADJUST
    TRAIN = False
    PRINT_HISTORY = False
    LEARNING_RATE = 1e-4
    VALIDATION_SPLIT = 0.2
    EPOCHS = 1

    # Get datasets and choose whether to generate a new model or load a model
    X_dataset, Y_dataset = util.get_dataset()
    conv_model = get_model(X_dataset, Y_dataset, lr=LEARNING_RATE, new=False)
    
    # TODO: This can be its own function
    if TRAIN:
        history_conv = conv_model.fit(X_dataset, Y_dataset, 
                                    validation_split=VALIDATION_SPLIT, 
                                    epochs=EPOCHS, 
                                    batch_size=16)

    # Fit the data and get history of the model over time, if specified
    if PRINT_HISTORY:
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

    # Testing the model
    plates = util.files_in_folder(PLATE_DIR)
    print("Testing:", plates[300])
    plate_to_test = plates[300]

    predict_plate(plate_to_test, conv_model)


if __name__ == '__main__':
    main()
