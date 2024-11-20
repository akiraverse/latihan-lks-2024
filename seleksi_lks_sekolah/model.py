import pandas as pd 
import numpy as np 





# making euclidean distance function
# point1 will represent the testing data
# point2 will represent the training data

class irisPrediction:
	def __init__(self, X_train, y_train, X_test, y_test, k):
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test
		self.k = k

	def euclidean_distance(self, point1, point2):
		return np.sqrt(np.sum(point1 - point2) ** 2)

    
    # get the top 'k' nearest neighbors
    # test_instance will be related to each data point in X_test

	def get_neighbors(self, X_train, y_train, test_instance, k):
		distances = [] 
		
		for i in range(len(self.X_train)): # looping based on how many rows in X_train
			dist = self.euclidean_distance(test_instance, self.X_train[i]) 
			#calculationg each test_instance data point with all the X_train data points
			
			distances.append((dist, self.y_train[i]))
			#inserting the distance and the target (species)
			#because both of these will be used in the prediction function
			
		distances.sort(key=lambda x:x[0])
		#sorting the distance by ascending

		neighbors = [distances[i][1] for i in range(k)]
		# assigning the 'k' nearest neighbors 
		# example if k = 5, meaning we are getting the top 5 closest neighbors
			
		return neighbors
		# returning the 'neighbors' array 


	def predict(self, X_train, y_train, X_test, k):
		predictions = []
		
		for test_instance in X_test: #looping the X_test to get the k neighbors of each test data point
			neighbors = self.get_neighbors(self.X_train, self.y_train, test_instance, k)
			#get the k neighbors of the test_instance
			
			prediction_species = max(set(neighbors), key=neighbors.count)
			#get the most frequent species out of all the 'k' neighbors
			
			predictions.append(prediction_species)
			#inserting the most frequent species to the predict 
			
		return predictions
		#returning the predict list




	# Making accuracy prediction by percentage
	def accuracy(self, y_true, y_pred):
		correct = np.sum(y_true == y_pred)
		acc = correct / len(y_true) * 100
		return f'The accuracy is: {acc:.2f}%'



	
	def predict_single(self, X_train, y_train, new_instance, k):
		# Get the neighbors for this new instance
		neighbors = self.get_neighbors(self.X_train, self.y_train, new_instance, k)
		
		# Determine the most frequent species among neighbors (majority voting)
		predicted_species = max(set(neighbors), key=neighbors.count)
		return predicted_species



def splitting(data, ratio, target):
	data = data.sample(frac=1)
	
	total_rows = data.shape[0] #picking the rows
	train_size = int(ratio * total_rows)

	train = data[0:train_size]
	test = data[train_size:]

	# training dataset assigning
	X_train, y_train = np.array(train.drop(columns=[target])), np.array(train[target])

	# testing dataset asigning
	X_test, y_test = np.array(test.drop(columns=[target])), np.array(test[target])	

	return X_train, y_train, X_test, y_test

