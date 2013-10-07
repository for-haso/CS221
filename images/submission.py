import collections
import numpy as np

def distance(p1, p2):
	v = p1 - p2
	# we want the norm squared, so just return the dot prod
	return np.dot(v,v)

############################################################
# Problem 2
def runKMeans(k,patches,maxIter):
	"""
	Runs K-means to learn k centroids, for maxIter iterations.

	Args:
	k - number of centroids.
	patches - 2D numpy array of size patchSize x numPatches
	maxIter - number of iterations to run K-means for

	Returns:
	centroids - 2D numpy array of size patchSize x k
	"""
	# This line starts you out with randomly initialized centroids in a matrix 
	# with patchSize rows and k columns. Each column is a centroid.
	centroids = np.random.randn(patches.shape[0],k)

	numPatches = patches.shape[1]
	
	patches = np.transpose(patches)
	centroids = np.transpose(centroids)
	for i in range(maxIter):
		# BEGIN_YOUR_CODE (around 19 lines of code expected)
		centroid_points = dict()
		for j in range(numPatches):
			centroid = np.argmin([distance(patches[j], centroids[x]) for x in range(k)]) 

			if centroid in centroid_points:
				centroid_points[centroid].append(patches[j])
			else:
				centroid_points[centroid] = list()
				centroid_points[centroid].append(patches[j])

		for j in range(k):
			centroids[j] = 1.0 * np.sum(centroid_points[j], axis=0) / len(centroid_points[j])

		# END_YOUR_CODE
	return np.transpose(centroids)

############################################################
# Problem 3

def extractFeatures(patches,centroids):
	"""
	Given patches for an image and a set of centroids, extracts and return
	the features for that image.

	Args:
	patches - 2D numpy array of size patchSize x numPatches
	centroids - 2D numpy array of size patchSize x k

	Returns:
	features - 2D numpy array with new feature values for each patch
	of the image in rows, size is numPatches x k
	"""
	k = centroids.shape[1]
	numPatches = patches.shape[1]
	features = np.empty((numPatches,k))

	# BEGIN_YOUR_CODE (around 9 lines of code expected)
	raise Exception("Not implemented yet")
	# END_YOUR_CODE
	return features

############################################################
# Problem 4a

def logisticGradient(theta,featureVector,y):
	"""
	Calculates and returns gradient of the logistic loss function with
	respect to parameter vector theta.

	Args:
	theta - 1D numpy array of parameters
	featureVector - 1D numpy array of features for training example
	y - label in {0,1} for training example

	Returns:
	1D numpy array of gradient of logistic loss w.r.t. to theta
	"""
	# BEGIN_YOUR_CODE (around 2 lines of code expected)
	raise Exception("Not implemented yet")
	# END_YOUR_CODE

############################################################
# Problem 4b

def hingeLossGradient(theta,featureVector,y):
	"""
	Calculates and returns gradient of hinge loss function with
	respect to parameter vector theta.

	Args:
	theta - 1D numpy array of parameters
	featureVector - 1D numpy array of features for training example
	y - label in {0,1} for training example

	Returns:
	1D numpy array of gradient of hinge loss w.r.t. to theta
	"""
	# BEGIN_YOUR_CODE (around 6 lines of code expected)
	raise Exception("Not implemented yet")
	# END_YOUR_CODE
