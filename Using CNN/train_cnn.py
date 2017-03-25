'''
Deep Learning Programming Assignment 2
--------------------------------------
Name:Vysyaraju Nayan Raju
Roll No.:14MA20049

======================================
Complete the functions in this file.
Note: Do not change the function signatures of the train
and test functions
'''
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import model_from_json
from keras.utils import np_utils
import shutil
import requests
from keras import backend as K
K.set_image_dim_ordering('th')

def createmodel():
    num_classes = 10
    # create model
    model = Sequential()
    # model.add(Convolution2D(32, 5, 5, border_mode='valid', input_shape=(1, 28, 28), activation='relu'))
    model.add(Convolution2D(28 , 5, 5, input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train(trainX, trainY):
    '''
    Complete this function.
    '''
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    trX = trainX.reshape(trainX.shape[0], 1, trainX.shape[1], trainX.shape[2]).astype('float32')
    trX = trX / 255
    # testX = testX / 255
    # one hot encode outputs
    trY = trainY
    trY = np_utils.to_categorical(trY)
    # testY = np_utils.to_categorical(testY)
    # num_classes = testY.shape[1]
    # build the model
    model = createmodel()
    # Fit the model
    model.fit(trX, trY,  nb_epoch=10, batch_size=32, verbose=1)
    cnn_model_json = model.to_json()
    with open("cnn_model.json", "w") as json_file:
        json_file.write(cnn_model_json)
    # serialize weights to HDF5
    model.save_weights("cnn_model.h5")
    print("********Saved model**********")

def test(testX):
    '''
    Complete this function.
    This function must read the weight files and
    return the predicted labels.
    The returned object must be a 1-dimensional numpy array of
    length equal to the number of examples. The i-th element
    of the array should contain the label of the i-th test
    example.
    '''

    x = input("\nDo you want to download weights deduced before from online or use the weights deduced now, (YES(TO DOWNLOAD) = 1 / NO = 0): ")

    if(x==1):
        print("******** Downloading Model **********")
        url = 'https://raw.githubusercontent.com/v-nayanraju/DeepLearning/master/ProgAssign3/cnn_model.h5'
        response = requests.get(url, stream=True)
        with open('cnn_model.h5', 'w') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        del response
        url = 'https://raw.githubusercontent.com/v-nayanraju/DeepLearning/master/ProgAssign3/cnn_model.json'
        response = requests.get(url, stream=True)
        with open('cnn_model.json', 'w') as out_file:
            out_file.write(response.text)
        del response



    # load json and create model
    json_file = open('cnn_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("cnn_model.h5")

    print ("***********Predicting************")
    testX=testX.reshape(-1,1, 28, 28)
    testX=testX/255
    ytest=loaded_model.predict(testX, batch_size=20 , verbose=0)
    ytest=ytest.reshape(-1, 10)
    ypreds=np.argmax(ytest, axis=1)
    return ypreds

    # return np.zeros(testX.shape[0])

