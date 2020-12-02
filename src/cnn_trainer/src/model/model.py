import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

import util as util


# Save a Keras model
def save_model(model, name):
    model.save('keras/' + name)


# Load a Keras model
def load_model(name):
    return models.load_model('keras/' + name)


# Generate completely new training model
def generate_model(lr, model_type):

    if model_type == 0:
        output_size = 26
    else:
        output_size = 10

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 105, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(output_size, activation='softmax'))

    # Set Learning Rate and Compile Model
    model.compile(loss='categorical_crossentropy',
                    optimizer=optimizers.RMSprop(lr=lr),
                    metrics=['acc'])

    return model


# Get a model either by generating a new one or load from local
def get_model(lr=1e-4, model_type=0, new=False):  

    if model_type == 0:
        name = 'letter_model'
    else:
        name = 'number_model'

    if new:
        print("Compiling new model.")
        model = generate_model(lr, model_type)
    else:
        print("Loading local model " + name)
        model = load_model(name)

    model.summary()

    return model


def train_model(model, X_dataset, Y_dataset, vs, epochs, augment=True):

    print(X_dataset.shape)
    print(Y_dataset.shape)

    if augment:
        print("Augmenting data.")

        aug = ImageDataGenerator(
            shear_range=0.65,
            rotation_range=20,
            zoom_range=0.12,
            preprocessing_function=util.add_noise,
            brightness_range=[0.15,1.1],
            validation_split=vs
        )

        print("Visualizing IDG.")
        #visualize_idg(aug, X_dataset)

        print("Creating augmented datasets.")

        training_dataset = aug.flow(X_dataset, Y_dataset, subset='training')
        validation_dataset = aug.flow(X_dataset, Y_dataset, subset='validation')

        print('Training using generator.')

        history_conv = model.fit(
            training_dataset,
            batch_size=16,
            epochs=epochs,
            verbose=1,
            validation_data=validation_dataset,
        )

    else:
        history_conv = model.fit(X_dataset, Y_dataset, 
                            validation_split=vs,
                            verbose=1, 
                            epochs=epochs, 
                            batch_size=16
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


# Predict a plate using a model
def predict_plate(plate, model, model_type):
    imgs, vecs = util.process_plate(plate, model_type)
    dataset = np.array(imgs)

    chars = []
    true = []

    if model_type == 0:
        for i in range(2):
            # plt.imshow(dataset[i])
            # plt.show()

            image = np.expand_dims(dataset[i], axis=0)

            y_true = vecs[i]
            index_true = np.argmax(y_true)

            y_predicted = model.predict(image)[0]
            index_predicted = np.argmax(y_predicted)

            print("Confidence for: ", util.index_to_val(index_true))
            print(y_predicted)

            true.append(util.index_to_val(index_true))
            chars.append(util.index_to_val(index_predicted))

        print("Actual:", true)
        print("Predicted:", chars) 

    if model_type == 1:
        for i in range(2):
            # plt.imshow(dataset[i])
            # plt.show()

            image = np.expand_dims(dataset[i], axis=0)

            y_true = vecs[i]
            index_true = np.argmax(y_true)

            y_predicted = model.predict(image)[0]
            index_predicted = np.argmax(y_predicted)

            print("Confidence for: ", index_true)
            print(y_predicted)

            true.append(index_true)
            chars.append(index_predicted)

        print("Actual:", true) 
        print("Predicted:", chars) 


# Predict the plates in the test set
def predict_test_set(plate, model, model_type):
    imgs = util.process_homographic_plate(plate, model_type)
    dataset = np.array(imgs)

    chars = []

    if model_type == 0:
        for i in range(2):
            # plt.imshow(dataset[i])
            # plt.show()

            image = np.expand_dims(dataset[i], axis=0)

            y_predicted = model.predict(image)[0]
            index_predicted = np.argmax(y_predicted)

            print("Confidence")
            print(y_predicted)
            print("\n")

            chars.append(util.index_to_val(index_predicted))

        print("Actual:", plate) 
        print("Predicted:", chars) 

    if model_type == 1:
        for i in range(2):
            # plt.imshow(dataset[i])
            # plt.show()

            image = np.expand_dims(dataset[i], axis=0)

            y_predicted = model.predict(image)[0]
            index_predicted = np.argmax(y_predicted)

            print("Confidence")
            print(y_predicted)
            print("\n")
            
            chars.append(index_predicted)

        print("Actual:", plate) 
        print("Predicted:", chars) 


def main():
    # PARAMETERS TO ADJUST
    TRAIN = True
    RESET_MODEL = False # BE CAREFUL WITH THIS.
    PREDICT = True
    AUGMENT = True

    # 0 for LETTER_MODEL, 1 for NUMBER_MODEL
    MODEL_TYPE = 0

    # Constants
    LEARNING_RATE = 1e-4

    # Letter model parameters.
    EPOCHS_1 = 2
    VS_1 = 0.2

    # Number model parameters.
    EPOCHS_2 = 2
    VS_2 = 0.2

    # Generate model or retrieve model
    model = get_model(lr=LEARNING_RATE, model_type=MODEL_TYPE, new=RESET_MODEL)
    
    if MODEL_TYPE == 0: # This corresponds to the model for plates
        if TRAIN:
            X_dataset, Y_dataset = util.get_training_dataset(MODEL_TYPE) 

            model = train_model(model,
                X_dataset,
                Y_dataset,
                VS_1,
                EPOCHS_1,
                augment=AUGMENT)

            save_model(model, 'letter_model')

        # Predict a plate if specified
        if PREDICT:
            plates = util.files_in_folder(util.PLATE_DIR)
            plate_to_test = plates[16]
            print("Testing ", plate_to_test)
            predict_plate(plate_to_test, model, MODEL_TYPE)

            # Predict from test set
            print("Testing from test set")
            test_plates = util.files_in_folder(util.TEST_PATH)
            for p in test_plates:
                predict_test_set(p, model, MODEL_TYPE)

    elif MODEL_TYPE == 1: # This corresponds to the model for numbers
        if TRAIN:
            X_dataset, Y_dataset = util.get_training_dataset(MODEL_TYPE)
            
            #for i,x in enumerate(X_dataset):
                # plt.imshow(x)
                # plt.show()
                # print(Y_dataset[i])
            
            model = train_model(model, 
                X_dataset,
                Y_dataset,
                VS_2,
                EPOCHS_2,
                augment=AUGMENT
                )

            save_model(model, 'number_model')

        if PREDICT:
            plates = util.files_in_folder(util.PLATE_DIR)
            plate_to_test = plates[0]
            print("Testing ", plate_to_test)
            predict_plate(plate_to_test, model, MODEL_TYPE)

            # Predict from test set
            print("Testing from test set")
            test_plates = util.files_in_folder(util.TEST_PATH)
            for p in test_plates:
                predict_test_set(p, model, MODEL_TYPE)


    else: # ur an idiot
        print("pick a model please.")


if __name__ == '__main__':
    main()
