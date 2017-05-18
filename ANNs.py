#!usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import division
import numpy as np
import math

np.random.seed(1234)

# definition of sigmoid funtion
# numpy.exp work for arrays.
def sigmoid(x):
	return 1 / (1 + np.exp(-x))


# definition of sigmoid derivative funtion
# input must be sigmoid function's result
def sigmoid_output_to_derivative(result):
	return result*(1-result)


# init training set
def getTrainingSet(nameOfSet):

	setDict = {
		"sin": getSinSet(),
		"line": getLineSet(),
		}

	return setDict[nameOfSet]

def getSinSet():

	x = 3.14 * np.random.rand(1)
	# x = 6.2 * np.random.rand(1) - 3.14
	x = x.reshape(1,1)
	# y = np.array([5 *x]).reshape(1,1)
	# y = np.array([math.sin(x)]).reshape(1,1)
	y = np.array([math.sin(x),1]).reshape(1,2)
	return x, y


def getLineSet():

	x = 1 * np.random.rand(4) - 0.5
	x = x.reshape(1,4)
	k = np.array([-43.83764134, -15.94397998,  34.92887039, -27.07145086])
	k = k.reshape(4,1)
	y = np.dot(x, k)

	return x, y


def getW(synapse, delta):

	resultList = []

	# 遍历隐藏层每个隐藏单元对每个输出的权值，比如8个隐藏单元，每个隐藏单元对两个输出各有2个权值
	for i in range(synapse.shape[0]):

		resultList.append(
			(synapse[i,:] * delta).sum()
			)

	resultArr = np.array(resultList).reshape(1, synapse.shape[0])

	return resultArr


def getT(delta, layer):

	result = np.dot(layer.T, delta)
	return result


def backPropagation(trainingExamples, etah, input_dim, output_dim, hidden_dim, hidden_num, convergence_condition):

	# 可行条件
	if hidden_num < 1:
		print("隐藏层数不得小于1")
		return

	# 初始化网络权重矩阵，这个是核心
	synapseList = []
	# 输入层与隐含层1
	synapseList.append(2*np.random.random((input_dim,hidden_dim)) - 1)
	# 隐含层1与隐含层2, 2->3,,,,,,n-1->n
	for i in range(hidden_num-1):
		synapseList.append(2*np.random.random((hidden_dim,hidden_dim)) - 1)
	# 隐含层n与输出层
	synapseList.append(2*np.random.random((hidden_dim,output_dim)) - 1)

	iCount = 0
	lastErrorMax = 99999

	# while True:
	for i in range(10000):
		
		errorMax = 0

		for x, y in trainingExamples:

			iCount += 1
			layerList = []

			# 正向传播
			layerList.append(
				sigmoid(np.dot(x,synapseList[0]))
				)
			for j in range(hidden_num):
				layerList.append(
					sigmoid(np.dot(layerList[-1],synapseList[j+1]))
					)


			# 对于网络中的每个输出单元k，计算它的误差项
			deltaList = []
			layerOutputError = y - layerList[-1]
			# 收敛条件
			errorMax = layerOutputError.sum() if layerOutputError.sum() > errorMax else errorMax

			deltaK = sigmoid_output_to_derivative(layerList[-1]) * layerOutputError
			deltaList.append(deltaK)

			iLength = len(synapseList)
			for j in range(hidden_num):
				w = getW(synapseList[iLength - 1 - j], deltaList[j])
				delta = sigmoid_output_to_derivative(layerList[iLength - 2 - j]) * w
				deltaList.append(delta)


			# 更新每个网络权值w(ji)
			for j in range(len(synapseList)-1, 0, -1):
				t = getT(deltaList[iLength - 1 -j], layerList[j-1])
				synapseList[j] = synapseList[j] + etah * t

			t = getT(deltaList[-1], x)
			synapseList[0] = synapseList[0] + etah * t

		print("最大输出误差:")
		print(errorMax)

		if abs(lastErrorMax - errorMax) < convergence_condition:

			print("收敛了")
			print("####################")
			break
		lastErrorMax = errorMax			


	# 测试训练好的网络
	for i in range(5):
		xTest, yReal = getSinSet()
		# xTest, yReal = getLineSet()
		
		layerTmp = sigmoid(np.dot(xTest,synapseList[0]))
		for j in range(1, len(synapseList), 1):
			layerTmp = sigmoid(np.dot(layerTmp,synapseList[j]))

		yTest = layerTmp
		print("x:")
		print(xTest)
		print("实际的y:")
		print(yReal)
		print("神经元网络输出的y:")
		print(yTest)
		print("最终输出误差:")
		print(np.abs(yReal - yTest))
		print("#####################")

	# test in training set
	precision = 0
	for x, y in trainingExamples:

		layerTmp = sigmoid(np.dot(x,synapseList[0]))
		for j in range(1, len(synapseList), 1):
			layerTmp = sigmoid(np.dot(layerTmp,synapseList[j]))
		yTest = layerTmp

		if abs(yTest[0][0] - y[0][0]) < 0.1:
			precision += 1

	print("precision on training set")
	print(str(precision / len(trainingExamples) * 100) + "%")


	print("迭代次数:")
	print(iCount)


if __name__ == '__main__':

	import datetime
	tStart = datetime.datetime.now()

	# 使用什么样的训练样例
	nameOfSet = "sin"
	# nameOfSet = "line"
	x, y = getTrainingSet(nameOfSet)
	# setting of parameters
	# 这里设置了学习速率。
	etah = 0.25
	# 隐藏层数
	hidden_num = 2
	# 网络输入层的大小
	input_dim = x.shape[1]
	# 隐含层的大小
	hidden_dim = 100
	# 输出层的大小
	output_dim = y.shape[1]
	# 收敛条件
	convergence_condition = 0.001
	
	# 构建训练样例
	trainingExamples = []
	for i in range(10000):
		x, y = getTrainingSet(nameOfSet)
		trainingExamples.append((x, y))

	# 开始用反向传播算法训练网络
	backPropagation(trainingExamples, etah, input_dim, output_dim, hidden_dim, hidden_num, convergence_condition)

	tEnd = datetime.datetime.now()

	print("time cost:")
	print(tEnd - tStart)