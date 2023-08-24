# Instituto Balseiro
# Redes Neuronales y Aprendizaje Profundo para Visión Artificial - 2019
# Alumno:  Enrique Nicanor Mariotti
# Carrera: Maestria en Cs de la Ingenieria

import numpy as np
import sys

from .activations import Sigmoid, ReLU, Tanh, Linear

class Layer():
	def __init__(self):
		pass

	def layer_name(self):
		return self.__class__.__name__

	def set_input_shape(self, shape):
		self.input_shape = shape

	def parameters(self):
		return 0

	def forward(self, layer_input, x):
		pass
	
	def backward(self, grad_acc, optimizer):
		pass

	def output_shape(self):
		pass

class Input(Layer):
	def __init__(self, input_shape):
		self.input_shape = input_shape

	def forward(self, layer_input, x):
		self.layer_input = layer_input
		return self.layer_input

	def output_shape(self):
		return self.input_shape

class Dense(Layer):
	def __init__(self, n_units, input_shape=None, input_stack=False, trainable=True):
		# Shaping
		self.n_units = n_units
		self.input_shape = input_shape
		self.input_stack = input_stack 
		
		# Parameters
		self.trainable = trainable
		self.layer_input = None
		self.w = None
		self.b = None

	def initialize(self):
		fan_in = self.input_shape[0]
		fan_out = self.n_units
		self.w = np.random.rand(self.n_units, self.input_shape[0]) * (2.0/np.sqrt(fan_in+fan_out))
		self.b = np.random.rand(self.n_units, 1) * (2.0/np.sqrt(fan_in+fan_out))

	def parameters(self):
		return np.prod(self.w.shape) + np.prod(self.b.shape)

	def forward(self, layer_input, x):
		# Stack if input_stack
		if self.input_stack:
			self.layer_input = np.vstack((layer_input, x))
		else:
			self.layer_input = layer_input
		score = np.dot(self.w, self.layer_input) + self.b
		return score

	def backward(self, grad_acc, optimizer):
		if self.trainable:
			# Weights gradient =  input.T  ⋅ grad_acc # Bias gradiente = 1 ⋅ grad_acc
			grad_w = np.dot(grad_acc, self.layer_input.T)			
			grad_b = np.sum(grad_acc, axis=1, keepdims=True)

			# Using the foward weights, calculate the new accumulated gradient 
			grad_acc = np.dot(self.w.T, grad_acc)

			# Update
			self.w = optimizer.update(self.w, grad_w)
			self.b = optimizer.update(self.b, grad_b)
			
			# Split if input_stack
			if self.input_stack:
				grad_acc = grad_acc[:self.n_units,:]
		return grad_acc

	def output_shape(self):
		return (self.n_units,)

class Activation(Layer):
	def __init__(self, function_name):
		activation_dic = { 'relu': ReLU, 'sigmoid': Sigmoid, 'tanh': Tanh, 'linear': Linear}
		self.activation_function = activation_dic[function_name]()

	def layer_name(self):
		return 'Activation ({})'.format(self.activation_function.__class__.__name__)

	def forward(self, layer_input, x):
		self.layer_input = layer_input
		return self.activation_function(self.layer_input)

	def backward(self, grad_acc, optimizer):
		return grad_acc * self.activation_function.gradient(self.layer_input)

	def output_shape(self):
		return self.input_shape