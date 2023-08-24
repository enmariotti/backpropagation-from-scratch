import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # {'0', '1', '2', '3'}
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['KMP_WARNINGS'] = 'off'

import time
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt

from src.models import NeuralNetwork
from src.optimizers import SGD
from src.layers import Input, Dense, Activation

from keras.datasets import cifar10, mnist

def data_encoder(y, n_labels):
	y_encoded = np.eye(n_labels)[y]
	y_encoded = y_encoded.astype('float32')  
	return y_encoded

def cifar10_data():
	(x_train, y_train), (x_test, y_test) = cifar10.load_data()
	# Data type 
	x_train = x_train.astype('float32') / 255.0
	x_test = x_test.astype('float32') / 255.0
	# Data preprocessing 
	mean_image = np.mean(x_train, axis=0)
	x_train -= mean_image
	x_test -= mean_image
	# Labels and classes
	n_labels = 10
	# Train & Test features
	x_train = x_train.reshape(x_train.shape[0],np.prod(x_train.shape[1:])).T 
	x_test = x_test.reshape(x_test.shape[0],np.prod(x_test.shape[1:])).T
	# Train labels 
	y_train = y_train.reshape(-1)
	y_train_encoded = data_encoder(y_train, n_labels).T
	# Test labels 
	y_test = y_test.reshape(-1)
	y_test_encoded = data_encoder(y_test, n_labels).T
	return (x_train, y_train_encoded), (x_test, y_test_encoded)

def print_history(history, name, test_plot=True):
	# Model plot		
	fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,10))
	plt.subplot(1, 2, 1)
	plt.plot(history['train_acc'], linewidth=3, label='Train Accuracy')
	if test_plot:
		plt.plot(history['test_acc'], linewidth=3, label='Test Accuracy')
	plt.xlabel('Number of epochs', fontsize=14)
	plt.ylabel('Accuracy', fontsize=14)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.legend(fontsize=14)
	plt.subplot(1, 2, 2)
	plt.plot(history['train_loss'], linewidth=3, label='Train Accuracy')
	if test_plot:
		plt.plot(history['test_loss'], linewidth=3, label='Test Accuracy')
	plt.xlabel('Number of epochs', fontsize=14)
	plt.ylabel('Loss', fontsize=14)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.legend(fontsize=14)
	# Save plot
	fig.savefig('./pr_{}_metric_loss.png'.format(name), bbox_inches='tight')
	print('Model plot saved in pwd')  
	return None	 

if __name__ == '__main__':

	# Parsers & sub-parsers
	parser = argparse.ArgumentParser()
	parser.add_argument('-p', '--problem', dest='problem', required=True, type=str, choices={'3','4','5','6','7','8'}, help='Problem number in {3,4,5,6,7,8}')
	
	# Dictionary of parsers
	args = parser.parse_args()

	# Fix random seed
	np.random.seed(1988)

	# problem command
	if args.problem=='3':
		# Load pre-process data
		(x_train, y_train), (x_test, y_test) = cifar10_data()

		# Optimizer
		SGD = SGD(learning_rate=0.001, reg_strength=0.01)
		
		# Models
		history = {}

		# Model #3
		model = NeuralNetwork(optimizer=SGD, loss='mse', metric='accuracy')
		model.add(Input(input_shape=(x_train.shape[0],)))
		model.add(Dense(100))
		model.add(Activation('tanh'))
		model.add(Dense(10))
		model.add(Activation('linear'))
		model.summary()

		#Model Fit
		start_time = time.time()
		history = model.fit(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, epochs=50, batch_size=32)
		end_time = time.time()
		print('\nModel elapsed training time: {:.5f} seconds'.format(end_time-start_time))

		# Plot
		print_history(history, args.problem)

	if args.problem=='4':
		# Load pre-process data
		(x_train, y_train), (x_test, y_test) = cifar10_data()

		# Optimizer
		SGD = SGD(learning_rate=0.001, reg_strength=0.01)
		
		# Models
		history = {}

		# Model #3
		model = NeuralNetwork(optimizer=SGD, loss='cce', metric='accuracy')
		model.add(Input(input_shape=(x_train.shape[0],)))
		model.add(Dense(100))
		model.add(Activation('tanh'))
		model.add(Dense(10))
		model.add(Activation('linear'))
		model.summary()

		#Model Fit
		start_time = time.time()
		history = model.fit(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, epochs=50, batch_size=32)
		end_time = time.time()
		print('\nModel elapsed training time: {:.5f} seconds'.format(end_time-start_time))

		# Plot
		print_history(history, args.problem)

	if args.problem=='5':
		# Load pre-process data
		(x_train, y_train), (x_test, y_test) = cifar10_data()

		# Optimizer
		SGD = SGD(learning_rate=0.001, reg_strength=0.01)
		
		# Models
		history = {}

		# Model #3
		model = NeuralNetwork(optimizer=SGD, loss='cce', metric='accuracy')
		model.add(Input(input_shape=(x_train.shape[0],)))
		model.add(Dense(100))
		model.add(Activation('relu'))
		model.add(Dense(10))
		model.add(Activation('sigmoid'))
		model.summary()

		#Model Fit
		start_time = time.time()
		history = model.fit(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, epochs=50, batch_size=32)
		end_time = time.time()
		print('\nModel elapsed training time: {:.5f} seconds'.format(end_time-start_time))

		# Plot
		print_history(history, args.problem)

	if args.problem=='6':
		# Create data
		x_train = np.array([[1, 1],[1, -1],[-1, 1],[-1, -1]]).T		
		y_train = np.prod(x_train, axis=0)
		y_train = y_train[np.newaxis, :]
		
		# Optimizer
		SGD = SGD(learning_rate=0.5, reg_strength=0.01)
		
		# Models
		models = {}
		history = {}

		# Model A
		model_A = NeuralNetwork(optimizer=SGD)
		model_A.add(Input(input_shape=(2,)))
		model_A.add(Dense(2))
		model_A.add(Activation('tanh'))
		model_A.add(Dense(1))
		model_A.add(Activation('tanh'))
		model_A.summary()
		models['A'] = model_A

		# Model B
		model_B = NeuralNetwork(optimizer=SGD)
		model_B.add(Input(input_shape=(2,)))
		model_B.add(Dense(1))
		model_B.add(Activation('tanh'))
		model_B.add(Dense(1, input_stack=True))
		model_B.add(Activation('tanh'))
		model_B.summary()
		models['B'] = model_B

		# Fit models
		for key, model in models.items():
			# Model fit
			start_time = time.time()
			history[key] = model.fit(x_train=x_train, y_train=y_train, epochs=200, batch_size=4)
			end_time = time.time()
			print('\nModel {}, elapsed training time: {:.5f} seconds'.format(key, end_time-start_time))

			print('Expected output: {}'.format(y_train))
			print('NN output: {}'.format(model.predict(x_train)))

			# Plot
			print_history(history[key], args.problem+key, test_plot=False)
	
	if args.problem=='7':
		# Create data
		N = 5
		combinations_list = list(itertools.product([-1, 1], repeat=N))
		x_train = np.array(combinations_list).T
		y_train = np.prod(x_train, axis=0)
		y_train = y_train[np.newaxis, :]

		# Optimizer
		SGD = SGD(learning_rate=0.35, reg_strength=0.01)
		
		# Models
		history = {}

		# Model #3
		model = NeuralNetwork(optimizer=SGD)
		model.add(Input(input_shape=(N,)))
		model.add(Dense(N*2))
		model.add(Activation('tanh'))
		model.add(Dense(1))
		model.add(Activation('tanh'))
		model.summary()

		start_time = time.time()
		history = model.fit(x_train=x_train, y_train=y_train, epochs=1000, batch_size=32)
		end_time = time.time()
		print('\nModel elapsed training time: {:.5f} seconds'.format(end_time-start_time))

		# Model plot		
		print_history(history, args.problem, test_plot=False)

	if args.problem=='8':
		# Load pre-process data
		(x_train, y_train), (x_test, y_test) = cifar10_data()

		# Optimizer
		SGD = SGD(learning_rate=0.001, reg_strength=0.01)
		
		# Models
		history = {}

		# Model #3
		model = NeuralNetwork(optimizer=SGD, loss='cce', metric='accuracy')
		model.add(Input(input_shape=(x_train.shape[0],)))
		model.add(Dense(100))
		model.add(Activation('tanh'))
		model.add(Dense(100))
		model.add(Activation('tanh'))
		model.add(Dense(10))
		model.add(Activation('linear'))
		model.summary()

		start_time = time.time()
		history = model.fit(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, epochs=50, batch_size=32)
		end_time = time.time()
		print('\nModel elapsed training time: {:.5f} seconds'.format(end_time-start_time))

		# Model plot		
		print_history(history, args.problem)