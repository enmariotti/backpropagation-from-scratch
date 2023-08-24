import numpy as np

class Linear():
	def __call__(self, x):
		return x

	def gradient(self, x):
		return 1

class ReLU():
	def __call__(self, x):
		return np.where(x >= 0, x, 0)

	def gradient(self, x):
		return np.where(x >= 0, 1, 0)

class Tanh():
	def __call__(self, x):
		return np.tanh(x)

	def gradient(self, x):
		return 1 - np.power(np.tanh(x),2)

class Sigmoid():
	def __call__(self, x):
		return 1 / (1 + np.exp(-x))

	def gradient(self, x):
		return self.__call__(x) * (1 - self.__call__(x))

if __name__ == '__main__':
	pass