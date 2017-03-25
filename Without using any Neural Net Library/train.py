'''
Deep Learning Programming Assignment 1
--------------------------------------
Name:Vysyaraju Nayan Raju
Roll No.:14MA20049

======================================
Complete the functions in this file.
Note: Do not change the function signatures of the train
and test functions
'''
import numpy as np

def fwd_prop(trX,Weights,bias,Weights2,bias2):
	hidden_layer = np.maximum(0, np.dot(trX, Weights) + bias) # note, ReLU activation
	scores = np.dot(hidden_layer, Weights2) + bias2
	exp_scores = np.exp(scores)
	probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) 
	return (probs,scores,hidden_layer)

def train(trainX, trainY):
	'''
	Complete this function.
	'''
	trX = trainX.reshape(trainX.shape[0], trainX.shape[1]*trainX.shape[2]).astype(np.float)
	trY=trainY
	h = 100 
	K= 10
	Weights = 0.01 * np.random.randn(trX.shape[1],h)
	bias = np.zeros((1,h))
	Weights2 = 0.01 * np.random.randn(h,K)
	bias2 = np.zeros((1,K))

	step_size = 0.005
	reg = 1e-3 

	num_examples = trX.shape[0]
	for i in xrange(350):

		(probs,scores,hidden_layer)=fwd_prop(trX,Weights,bias,Weights2,bias2)
		corect_logprobs = -np.log(probs[range(num_examples),trY])
		data_loss = np.sum(corect_logprobs)/num_examples
		reg_loss = 0.5*reg*np.sum(Weights*Weights) + 0.5*reg*np.sum(Weights2*Weights2)
		loss = data_loss + reg_loss
		
		print "iteration %d: loss %f" % (i, loss)

		dscores = probs
		dscores[range(num_examples),trY] -= 1
		dscores /= num_examples

		dWeights2 = np.dot(hidden_layer.T, dscores)
		dbias2 = np.sum(dscores, axis=0, keepdims=True)
		dhidden = np.dot(dscores, Weights2.T)

		#print dhidden.shape,hidden_layer.shape
		dhidden[hidden_layer <= 0] = 0

		dWeights = np.dot(trX.T, dhidden)
		dbias = np.sum(dhidden, axis=0, keepdims=True)

		dWeights2 += reg * Weights2
		dWeights += reg * Weights

		Weights += -step_size * dWeights
		bias += -step_size * dbias
		Weights2 += -step_size * dWeights2
		bias2 += -step_size * dbias2

	np.save('Weights.npy',Weights)
	np.save('bias.npy',bias)
	np.save('Weights2.npy',Weights2)
	np.save('bias2.npy',bias2)



def test(testX):
	'''
	Complete this function.
	This function must read the Weight files and
	return the predicted labels.
	The returned object must be a 1-dimensional numpy array of
	length equal to the number of examples. The i-th element
	of the array should contain the label of the i-th test
	example.
	'''
	teX = testX.reshape(testX.shape[0], testX.shape[1]*testX.shape[2]).astype(np.float)
	Weights = np.load('Weights.npy')
	bias = np.load('bias.npy')
	Weights2 = np.load('Weights2.npy')
	bias2 = np.load('bias2.npy')
	(probs,scores,hidden_layer) = fwd_prop(teX,Weights,bias,Weights2,bias2)
	labels = np.argmax(probs,axis=1)


	return labels