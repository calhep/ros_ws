import numpy as np
import os

from matplotlib import pyplot as plt
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import plot_model

import util as util


# Save a Keras model
def save_model(model):
    model.save('keras/my_model')


# Load a Keras model
def load_model():
    return models.load_model('keras/my_model')


# Generate completely new training model
def generate_model(lr):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 105, 3)))
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


# Get a model either by generating a new one or load from local
def get_model(lr=1e-4, new=False):  

    if new:
        print("compiling new model")
        model = generate_model(lr)
    else:
        print("Loading local model")
        model = load_model()

    model.summary()

    return model


def train_model(model, X_dataset, Y_dataset, vs, epochs):
    history_conv = model.fit(X_dataset, Y_dataset, 
                                    validation_split=vs, 
                                    epochs=epochs, 
                                    batch_size=16)

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
    NEW_MODEL = False
    PREDICT = True

    LEARNING_RATE = 1e-4
    VALIDATION_SPLIT = 0.2
    EPOCHS = 3

    # Get datasets (A lot of plates)
    X_dataset, Y_dataset = util.get_dataset()

    # Generate model or retrieve model
    model = get_model(lr=LEARNING_RATE, new=NEW_MODEL)

    # Train the model if specified, always train if it's a new model.
    if TRAIN or NEW_MODEL:
        model = train_model(model, X_dataset, Y_dataset, VALIDATION_SPLIT, EPOCHS)
        save_model(model)

    # Predict a plate if specified
    if PREDICT:
        plates = util.files_in_folder(util.PLATE_DIR)
        plate_to_test = plates[159]

        print("Testing:", plate_to_test)
        predict_plate(plate_to_test, model)


if __name__ == '__main__':
    main()
