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
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import model_from_json
from keras.optimizers import SGD
from keras import backend as K
import shutil
import requests

K.set_image_dim_ordering('th')

# define baseline model
def baseline_model(trainX, trainY):
	num_pixels = trainX.shape[1]*trainX.shape[2]
	num_classes = 10
	# create model
	model = Sequential()
	model.add(Dense(28, input_dim=num_pixels, init='normal', activation='relu'))
	model.add(Dense(num_classes, init='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


def train(trainX, trainY):
	'''
	Complete this function.
	'''
	seed = 7
	np.random.seed(seed)
	trX = trainX.reshape(trainX.shape[0], trainX.shape[1]*trainX.shape[2]).astype(np.float)
	trY = trainY
	trX = trX/255
	trY = np_utils.to_categorical(trY)
	model = baseline_model(trainX,trainY)
	# Fit the model
	model.fit(trX, trY, nb_epoch=20, batch_size=32, verbose=1)
	dense_model = model.to_json()
	with open("dense_model.json", "w") as file:
		file.write(dense_model)
	# serialize weights to HDF5
	model.save_weights("dense_model.h5")
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
		url = 'https://raw.githubusercontent.com/v-nayanraju/DeepLearning/master/ProgAssign3/dense_model.h5'
		response = requests.get(url, stream=True)
		with open('dense_model.h5', 'w') as out_file:
			shutil.copyfileobj(response.raw, out_file)
		del response
		url = 'https://raw.githubusercontent.com/v-nayanraju/DeepLearning/master/ProgAssign3/dense_model.json'
		response = requests.get(url, stream=True)
		with open('dense_model.json', 'w') as out_file:
			out_file.write(response.text)
		del response




	# load json and create model
	json_file = open('dense_model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("dense_model.h5")

	print ("***********Predicting************")
	testX=testX.reshape(-1, 784)
	testX=testX/255
	ytest=loaded_model.predict(testX, batch_size=20	, verbose=0)
	ytest=ytest.reshape(-1, 10)
	ypreds=np.argmax(ytest, axis=1)
	return ypreds

	# return np.zeros(testX.shape[0])