#! usr/bin/env python3
# -*-coding=utf-8-*-
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import math

np.random.seed(1234)

class weakClassifer(object):
	def __init__(self):
		self.bestStump = {}

	def stumpClassify(self, X,featureIndex,threshVal,threshIneq):#just classify the data
		arr = np.ones(X.shape[0])
		if threshIneq == 'lt':
			arr[X[:,featureIndex] <= threshVal] = -1.0
		else:
			arr[X[:,featureIndex] > threshVal] = -1.0
		return arr


	def fit(self, X, Y, sampleWeight):
		"""Training weak classifier with input data with weight
		:param X: 2D-array in dimention m * n, m is number of feature, and n is quantity of data set
		:param Y: 1D-array in dimention n, n is quantity of data set, and every y of Y should in {-1, +1}
		:param sampleWeight: weight for each x in X
		"""
		featureNums = X.shape[1]

		# 步长, 最佳分类器模型
		numSteps = 1000.0; bestStump = {}
		minError = np.inf # 初始值设为正无穷大

		# 开始训练，遍历整个搜索空间
		for featureIndex in range(featureNums):
			rangeMin = X[:,featureIndex].min(); rangeMax = X[:,featureIndex].max();
			# print(rangeMin, rangeMax)
			stepSize = (rangeMax-rangeMin)/numSteps
			# print(stepSize)
			for threshVal in np.arange(rangeMin, rangeMax, stepSize):
				for inequal in ['lt', 'gt']:

					predictedVal = self.stumpClassify(X,featureIndex,threshVal,inequal)
					errArr = np.ones(X.shape[0])
					errArr[predictedVal == Y] = 0
					weightedError = np.dot(sampleWeight, errArr)

					if weightedError < minError:
						minError = weightedError
						bestStump['featureIndex'] = featureIndex
						bestStump['thresh'] = threshVal
						bestStump['ineq'] = inequal

		self.bestStump = bestStump


	def predict(self, X):
		arr = np.ones(X.shape[0])
		if self.bestStump['ineq'] == 'lt':
			arr[X[:,self.bestStump['featureIndex']] <= self.bestStump['thresh']] = -1.0
		else:
			arr[X[:,self.bestStump['featureIndex']] > self.bestStump['thresh']] = -1.0
		return arr


def getTrainingSample(rBig = 3, rSmall = 2, quantity = 100):
	"""Return: two demi cicle data, one big(r = 3) and one small(r = 2)
	:param rBig: big cicle with radius=3
	:param rSmall: small cicle with radius=2
	:param quantity: quantity of training data
	"""
	quantity = int(math.floor(quantity / 2))

	# generate big cicle data
	xBig = 2 * rBig * np.random.random(quantity) - rBig
	yBig = np.sqrt(
		rBig ** 2 - np.square(xBig)
		)
	yBig[::2] *= -1

	# plot big cicle data
	# plt.scatter(xBig, yBig, marker='.', color='k')

	# generate small cicle data
	xSmall = 2 * rSmall * np.random.random(quantity) - rSmall
	ySmall = np.sqrt(
		rSmall ** 2 - np.square(xSmall)
		)
	ySmall[::2] *= -1

	# plot small cicle data
	# plt.scatter(xSmall, ySmall, marker='*', color='r')
	# plt.show()

	# 2 dimention training data: [x1, y1], [x2, y2]
	trainingData = np.hstack((xBig.reshape(quantity,1), yBig.reshape(quantity,1)))
	trainingLabel = np.ones(quantity)

	trainingData = np.vstack((
		trainingData,
		(np.hstack((xSmall.reshape(quantity,1), ySmall.reshape(quantity,1))))
		))
	trainingLabel = np.hstack((
		trainingLabel, np.zeros(quantity) - 1
		))
	return trainingData, trainingLabel


class adaboost(object):
	def __init__(self, weakClassifer, nWeakClassifer=50, learningRate=1):
		"""Init internel params
		:param weakClassifer: class of weak classifier, it could be create new object
		:param nWeakClassifer: number of weak classifier
		"""
		self.weakClassifer = weakClassifer
		self.nWeakClassifer = nWeakClassifer
		self.learningRate = learningRate


	def _boost(self, iBoost, X, Y, sampleWeight):
		"""A single boost.
		:param iboost: int
			The index of the current boost iteration.
		:param X: {array-like, sparse matrix} of shape = [n_samples, n_features]
			The training input samples. Sparse matrix can be CSC, CSR, COO,
			DOK, or LIL. DOK and LIL are converted to CSR.
		:param Y: array-like of shape = [n_samples]
			The target values (class labels).
		sampleWeight : array-like of shape = [n_samples]
			The current sample weights.
		Returns
		-------
		sampleWeight : array-like of shape = [n_samples] or None
			The reweighted sample weights.
		alfa : float
			The weight for the current boost.
		error : float
			The classification error for the current boost.
		"""

		iWeakClassifer = self.weakClassifer()
		iWeakClassifer.fit(X, Y, sampleWeight=sampleWeight)

		yPredict = iWeakClassifer.predict(X)

		# Instances incorrectly classified
		incorrect = yPredict != Y

		# Error fraction
		error = np.mean(np.average(incorrect, weights=sampleWeight, axis=0))

		# Stop if classification is perfect
		if error <= 0:
			return sampleWeight, 1., 0., iWeakClassifer

		# Boost weight using multi-class AdaBoost SAMME alg
		alfa = self.learningRate * (
			np.log((1. - error) / max(error, 1e-16))
			)

		# Only boost the weights if I will fit again
		if iBoost != self.nWeakClassifer - 1:
			# Only boost positive weights
			sampleWeight *= np.exp(alfa * incorrect *
						((sampleWeight > 0) |
						(alfa < 0)))

		return sampleWeight, alfa, error, iWeakClassifer


	def fit(self, X, Y):
		"""Training adaboost classifier with X, Y
		:param X: 2D-array in dimention m * n, m is number of feature, and n is quantity of data set
		:param Y: 1D-array in dimention n, n is quantity of data set, and every y of Y should in {-1, +1}
		It return a adaboost classifier with a series of weak classifiers and a series of weitghs of weak classifiers
		"""

		sampleWeight = np.zeros(len(X)) + 1/len(X)
		self.classifiers = [] # classifier list [c1, c2, c3]
		self.alphas = np.zeros(self.nWeakClassifer, dtype=np.float) # alpha list [alpha1, alpha2, alpha3] weak classifier weights
		self.errors = np.ones(self.nWeakClassifer, dtype=np.float) # error list [e1, e2, ...] weak classifier errors

		for iBoost in range(self.nWeakClassifer):
			# Boosting step
			sampleWeight, alfa, error, classifier = self._boost(iBoost, X, Y, sampleWeight)

			self.alphas[iBoost] = alfa
			self.errors[iBoost] = error
			self.classifiers.append(classifier)

			if error == 0:
				break

			sampleWeightSum = np.sum(sampleWeight)
			# Stop if the sum of sample weights has become non-positive
			if sampleWeightSum <= 0:
				break

			if iBoost < self.nWeakClassifer - 1:
				# Normalize
				sampleWeight /= sampleWeightSum

		# update number of weak classifier of adaboost
		# this number could change when algorithm terminate before loop to end
		self.nWeakClassifer = len(self.classifiers)

		return self

	def predict(self, X):

		pred = np.zeros(X.shape[0])

		for iBoost in range(self.nWeakClassifer):
			classifier = self.classifiers[iBoost]
			alpha = self.alphas[iBoost]

			arr = classifier.predict(X)
			pred += arr * alpha
		pred[pred >= 0] = 1
		pred[pred < 0] = -1
		return pred


if __name__ == '__main__':

	# get training samples
	trainingData, trainingLabel = getTrainingSample(5, 4.5, 500)

	# init adaboost classifier
	adaboostclassifier = adaboost(weakClassifer, 100)

	# fit training data
	adaboostclassifier.fit(trainingData, trainingLabel)

	arr = adaboostclassifier.predict(trainingData)

	print(arr == trainingLabel)

	a = arr == trainingLabel
	print("accuracy:",
	a.sum() / len(a) * 100
	)

	# show result
	for i in range(trainingData.shape[0]):
		if arr[i] == 1:
			plt.scatter(trainingData[i][0], trainingData[i][1], marker='.', color='k')
		else:
			plt.scatter(trainingData[i][0], trainingData[i][1], marker='*', color='r')
	plt.show()