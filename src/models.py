import numpy as np
import random
import sys

from .optimizers import SGD
from .losses import MSE, CCE
from .metrics import accuracy
from .layers import Input, Dense, Activation

class NeuralNetwork():
	def __init__(self, optimizer, loss='mse', metric='accuracy'):
		# Create optimizer and losses
		losses_dic = {'mse': MSE, 'cce': CCE}
		metrics_dic = {'accuracy': accuracy}
		self.optimizer = optimizer
		self.loss = losses_dic[loss]()
		self.metric = metrics_dic[metric]()
		# Layers list
		self.layers = []

	def add(self, layer):
		# Dimsension matching
		if self.layers:
			if hasattr(layer, 'input_stack') and layer.input_stack:	
				layer.set_input_shape(shape=(self.layers[-1].output_shape()[0] + self.layers[0].output_shape()[0],))
			else:
				layer.set_input_shape(shape=self.layers[-1].output_shape())
		# Initialize weights
		if hasattr(layer, 'initialize'):
			layer.initialize()
		# Add layer to the network
		self.layers.append(layer)

	def fit(self, x_train, y_train, x_test=None, y_test=None, epochs=10, batch_size=16, shuffle_data=True):
		# Dimension
		self.epochs = epochs
		self.batch_size = batch_size
		self.shuffle_data = shuffle_data
		self.dim, self.n_samples = x_train.shape[0], x_train.shape[1]
		#Test data
		if x_test is None:
			x_test = x_train
			y_test = y_train
		# Labels and classes
		self.n_labels = np.max(np.unique(y_train)) + 1

		# Testn and Train History Log
		history = {}
		# Train stats
		history['train_loss'] = np.zeros(self.epochs)
		history['train_acc'] = np.zeros(self.epochs)
		# Test stats
		history['test_loss'] = np.zeros(self.epochs)
		history['test_acc'] = np.zeros(self.epochs)

		for e in range(self.epochs):
			# Stats over training batch
			history['train_loss'][e] = np.mean(self.loss(y_train, self.predict(x_train)))
			history['train_acc'][e] = self.metric(y_train, self.predict(x_train))

			# # Stats over test batch
			history['test_loss'][e] = np.mean(self.loss(y_test, self.predict(x_test)))
			history['test_acc'][e] = self.metric(y_test, self.predict(x_test))

			# # Print stats
			print('\n#Epoch: {:d}'.format(e))
			print('Train Loss: {:.5f}'.format(history['train_loss'][e]))
			print('Train Accuracy: {:.5f}'.format(history['train_acc'][e]))
			print('Test Loss: {:.5f}'.format(history['test_loss'][e]))
			print('Test Accuracy: {:.5f}'.format(history['test_acc'][e]))
						
			# Stochastic Gradient Descent
			self.mini_batches(x_train, y_train)
		return history

	def mini_batches(self, x, y):
		# Shuffle data
		if self.shuffle_data:
			idx = random.sample(range(self.n_samples), self.n_samples)
		else:
			idx = range(self.n_samples)
		x = x[:,idx]
		y = y[:,idx]
		# Gradient Descent with mini-batches
		for i in range(0, self.n_samples, self.batch_size):
			x_batch = x[:,i:self.batch_size+i]
			y_batch = y[:,i:self.batch_size+i]
			self.train_batch(x_batch, y_batch)

	def train_batch(self, x, y):
		y_score = self.feedforward(x)
		loss_grad = self.loss.gradient(y, y_score)
		self.backpropagate(loss_grad)

	def feedforward(self, x):
		layer_output = x
		for layer in self.layers:
			layer_output = layer.forward(layer_input=layer_output,x=x)
		return layer_output

	def backpropagate(self, loss_grad):
		grad_acc = loss_grad
		for layer in reversed(self.layers):
			grad_acc = layer.backward(grad_acc, self.optimizer)

	def predict(self, x):
		predicted_class = self.feedforward(x)
		return predicted_class

	def summary(self):		
		table_data = [["Layer Type", "Parameters", "Output Shape"]]
		total_parameters = 0
		for i, layer in enumerate(self.layers):
			print('\nLayer {}: {}'.format(i, layer.layer_name()))
			print('Layer {}: {} parameters'.format(i, layer.parameters()))
			print('Layer {}: {} output size'.format(i, layer.output_shape()))
			total_parameters += layer.parameters()
		print ('\nTotal Trainable Parameters: {}\n'.format(total_parameters))

if __name__ == '__main__':
	pass