import numpy as np

class MSE():

	def __call__(self, y, y_score):
		return 0.5 * np.power((y - y_score), 2)

	def gradient(self, y, y_score):
		return -(y - y_score)

class CCE():

	def __call__(self, y, y_score):
		softmax = self.softmax(y_score)
		loss = - np.log(softmax[y==1])
		return loss

	def softmax(self, scores):
		scores -= np.max(scores) # Normalization
		softmax = (np.exp(scores) / np.sum(np.exp(scores), axis=0))
		return softmax

	def gradient(self, y, y_score):
		softmax = self.softmax(y_score)		
		return softmax - y

if __name__ == '__main__':
	pass