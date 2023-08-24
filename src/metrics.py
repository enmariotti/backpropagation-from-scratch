import numpy as np

class accuracy():
	
	def __call__(self, y, y_score):
		if y.shape[0]==1:
			return np.mean(y == np.round(y_score))
		else:
			predicted_class = np.argmax(y_score, axis=0)
			real_class = np.argmax(y, axis=0)
			return np.mean(real_class == predicted_class)
		
if __name__ == '__main__':
	pass