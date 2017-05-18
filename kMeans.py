#!/usr/bin/env python3
# -*-conding:utf-8-*-

"""
K-means algorithm
~~~~~~~~~~~~~~~~~

by lty
"""
import numpy as np
from matplotlib import pyplot as plt

def getTrainingSet(dimension=3, quantity=10):
	"""Return a matrix of training set
	获取训练样例，每一条样例是矩阵的一行

	:param dimension: n of [x1, x2, ..., xn]
	:param quantity: quantity of training set
	"""

	if quantity < 2:
		raise Exception("quantity must >= 2")

	def getX():
		x = 10 * np.random.rand(dimension) - 5
		return x

	matrix = getX()
	for i in range(quantity - 1):
		matrix = np.vstack((matrix, getX()))

	return matrix


def normalization(matrix):
	"""归一化各个属性：避免取值大的属性对距离的影响高于取值范围小的属性
	矩阵的列对应一个属性 为每个属性归一化
	ai' = (ai - min(ai)) / (max(ai)-min(ai))
	example:
	X1 = {2, 1, 102} X2 = {1, 3, 2}
	X1' = {(2 - 1)/(2 - 1), (1 - 1)/(3  - 1), (102-2)/(102-2)}

	:param: matrix: all training set
	"""
	
	colomnNum = matrix.shape[1]
	for i in range(colomnNum):

		maxVal = max(matrix[:, i])
		minVal = min(matrix[:, i])
		denominator = maxVal - minVal

		tmpArr = matrix[:, i] - minVal
		matrix[:, i] = tmpArr / denominator

	return matrix


def centerInit(randomChoiseCenter, matrix, rawNum, k):
	"""Init centers point

	:param randomChoiseCenter: is init center points from training set
	:param matrix: if randomChoiseCenter == True, random choise center from training set
	:param rawNum: quantity of training set
	:param k: number of class
	"""

	if randomChoiseCenter == False:
		colomnNum = matrix.shape[1]
		centers = getTrainingSet(colomnNum, k)
		centers = normalization(centers)
		
	else:
		indexs = np.random.choice(rawNum, k)
		centers = matrix[indexs[0]]
		for index in range(1, len(indexs)):
			centers = np.vstack((centers, matrix[index]))
	return centers


def resultShow(matrix, k, centers, classifyArr):
	"""Show result only when dimention is 2

	:param matrix: training set
	:param k: number of class
	:param centers: center points result
	:param classifyArr: result of training set classification, len(classifyArr) == quantity of training set
	"""

	mark = [".",
		",",
		"o",
		"v",
		"^",
		"<",
		">",
		"1",
		"2",
		"3",
		"4",
		"8",
		"s",
		"p",
		"h",
		"H",
		"+",
		"x",
		"D",
		"d",
		"|",
		"_",
		"TICKLEFT",
		"TICKRIGHT",
		"TICKUP",
		"TICKDOWN",
		"CARETLEFT",
		"CARETRIGHT",
		"CARETUP",
		"CARETDOWN",]
	color=['r', 'k', 'b', [0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]]

	for iClassify in range(k):
		currentMatrix = matrix[np.where(classifyArr == iClassify)]
		plt.scatter(currentMatrix[:, 0], currentMatrix[:, 1], marker=mark[iClassify], color=color[iClassify])
	plt.scatter(centers[:, 0], centers[:, 1], marker="*", s=200)
	plt.show()
	return


def Kmeans(matrix, k=2, randomChoiseCenter=False):
	"""Main function of the algorithm K-means
	
	:param matrix: training set
	:param k: number of class
	:param randomChoiseCenter: is init center points from training set
	"""

	rawNum = matrix.shape[0]
	colomnNum = matrix.shape[1]

	centers = centerInit(randomChoiseCenter, matrix, rawNum, k)

	oldCenters = np.copy(centers)
	# 重复过程直到收敛
	# 收敛条件：10000次迭代或质心变化极小
	for i in range(10000):

		print("centers of mass:")
		print(centers)

		distanceList = []
		for center in centers:

			tmpDistance = matrix - center
			tmpDistance = np.power(tmpDistance, 2)

			distance = np.random.rand(rawNum).reshape(rawNum, 1)
			for j in range(rawNum):
				distance[j] = sum(tmpDistance[j, :])

			distanceList.append(distance)

		# 距离矩阵
		# 每行代表一个样本
		# 每列对应一个质心
		distanceMatrix = distanceList[0]
		for iPoint in range(1, len(distanceList)):
			distanceMatrix = np.hstack((distanceMatrix, distanceList[iPoint]))

		# 为每个训练样例计算距离最近的质心，即为每个训练样例分类
		clasifyResultList = []
		for iPoint in range(rawNum):
			iMinIndex = np.argmin(distanceMatrix[iPoint])
			clasifyResultList.append(iMinIndex)
		classifyArr = np.array(clasifyResultList)

		# K个分类中
		for iClassify in range(k):

			# 对于每个分类，筛选属于该分类的训练样例
			currentMatrix = matrix[np.where(classifyArr == iClassify)]

			# 重新计算质心位置
			iLength = currentMatrix.shape[0]
			if iLength == 0:
				centers = centerInit(randomChoiseCenter, matrix, rawNum, k)
				continue
			centers[iClassify] = np.sum(currentMatrix, axis=0) / iLength
			

		# 收敛条件
		if sum(sum(oldCenters-centers)) == 0.0:
			print("迭代次数: " + str(i + 1))
			break
		oldCenters = np.copy(centers)

	if colomnNum == 2 and k < 11:
		resultShow(matrix, k, centers, classifyArr)
	return clasifyResultList


if __name__ == '__main__':

	# dimension=2, quantity=10
	trainingSample = getTrainingSet(2, 30)
	# trainingSample = normalization(trainingSample)

	print("trainingSample:")
	print(trainingSample)
	# matrix=trainingSample, k=2
	clasifyResultList = Kmeans(trainingSample, 5, randomChoiseCenter=True)
	clasifyResultList = Kmeans(trainingSample, 5, randomChoiseCenter=True)
	clasifyResultList = Kmeans(trainingSample, 5, randomChoiseCenter=True)
	clasifyResultList = Kmeans(trainingSample, 5, randomChoiseCenter=True)
	clasifyResultList = Kmeans(trainingSample, 5, randomChoiseCenter=True)
	clasifyResultList = Kmeans(trainingSample, 5, randomChoiseCenter=True)
	clasifyResultList = Kmeans(trainingSample, 5, randomChoiseCenter=True)



	print(clasifyResultList)
	