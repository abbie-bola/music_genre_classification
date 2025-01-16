import json
import numpy as np 
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt

DATASET_PATH = "data.json"

def load_data(dataset_path):
    with open(dataset_path, "r") as fp: 
        data = json.load(fp)

    # convert lists into numpy arrays
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])
    return inputs, targets

def load_genres(dataset_path=DATASET_PATH):
    with open(dataset_path, "r") as fp: 
        data = json.load(fp)

    genres = np.array(data["mapping"])
    return genres

def prepare_datasets(test_size, validation_size):

    # load data
    inputs, targets = load_data(DATASET_PATH)
    
    # create train/test split 
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=test_size)

    # create the train/validation split
    inputs_train, inputs_validation, targets_train, targets_validation = train_test_split(inputs_train, targets_train, test_size=validation_size)

    # for a CNN, tensorflow expects a 3d array for each sample, currently inputs_train samples are 2d arrays 
    # 3d array -> (130, 13(mfcc), 1(depth))
    inputs_train = inputs_train[..., np.newaxis] # 4d array -> (num_samples, 130, 13, 1)
    inputs_validation = inputs_validation[..., np.newaxis]
    inputs_test = inputs_test[..., np.newaxis]

    return inputs_train, inputs_validation, inputs_test, targets_train, targets_validation, targets_test


def build_model(input_shape):

    """Generates CNN model
    param:
        input_shape (tuple): shape of input set 
    return: model (CNN model)"""

    # create model 
    model = keras.Sequential()
    
    # 1st conv layer  
    model.add(keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2,2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd conv layer 
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2,2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2,2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # flatten the output at feed it into dense layer 
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3)) # to prevent overfitting

    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model
    
def predict(model, inputs, targets):

    with open(DATASET_PATH, "r") as fp: 
        data = json.load(fp)

    genres = np.array(data["mapping"])
    
    # inputs -> 3d array: (130, 13, 1)
    inputs = inputs[np.newaxis, ...] # model expects 4d array, 4th dimention is no. of samples -> (1, 130, 13, 1)

    # prediction = [[0.1, 0.2, ....]] -> 10 values, indicating different probabilities for different genres
    prediction = model.predict(inputs) 

    # get the index with max value
    predicted_index = np.argmax(prediction, axis=1) 

    #predicted_genre = genres[predicted_index]

    #expected_genre = data.get("Mapping", targets)
    print("Expected index: {}, Predicted index:{}".format(targets, predicted_index))
    print("Expected genre: {}, Predicted genre:{}".format(genres[targets], genres[predicted_index]))


def plot_history(history):
    fig, axs = plt.subplots(2) 

    # create accuracy subplot 
    axs[0].plot(history.history["accuracy"], label="train accuracy") # the accuracy of the train set is stored in a dict called history and the key is history
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error subplot
    axs[1].plot(history.history["loss"], label="train error") # the accuracy of the train set is stored in a dict called history and the key is history
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()

if __name__ == "__main__":
    # create train, validation, and test sets
    inputs_train, inputs_validation, inputs_test, targets_train, targets_validation, targets_test = prepare_datasets(0.25, 0.2)

    # build the CNN 
    input_shape = (inputs_train.shape[1], inputs_train.shape[2], inputs_train.shape[3])
    model = build_model(input_shape)


    # compile the network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])


    # train the CNN 
    model.summary()
    
    history = model.fit(inputs_train, targets_train,
                  validation_data=(inputs_validation, targets_validation),
                  batch_size=32, epochs=30)


    # evaluate the CNN on the test set
    test_error, test_accuracy = model.evaluate(inputs_test, targets_test, verbose=1)
    print("Accuracy on test set is: {}".format(test_accuracy))

    # make predictions on a sample (inference)
    inputs = inputs_test[100]
    targets = targets_test[100]
    
    predict(model, inputs, targets)
    
    # plot accuracy/error for training and validation
    plot_history(history)
    
    
