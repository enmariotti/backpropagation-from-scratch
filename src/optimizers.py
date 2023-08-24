import numpy as np

class SGD():
	def __init__(self, learning_rate=0.07, reg_strength=0.01):
		self.learning_rate = learning_rate
		self.reg_strength = reg_strength

	def update(self, w, grad_w):
		reg_grad = self.reg_strength * w
		w_updated = w - self.learning_rate * (grad_w + reg_grad)
		return w_updated

if __name__ == '__main__':
	pass