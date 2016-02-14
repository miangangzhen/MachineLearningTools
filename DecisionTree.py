#!/usr/bin/env python3
# -*- encoding=utf-8 -*-
from math import log2

class DT(object):
	"""
		docstring for DT
		ID3 write by miangangzhen
	"""
	def __init__(self):
		super(DT, self).__init__()
		# self.id = 0
		self.treeList = []


	def train(self, trainSet):

		# preFunction
		# self.treeNodeList = []
		self.featureValueDict = self.getFeatureValueDict(trainSet)

		self.nodeSelectionRecursion(trainSet, [], [])


	def nodeSelectionRecursion(self, trainSet, labelList, labelUsedList):

		# 计算选择根节点
		maxGain = 0.0
		maxGainLable = ""
		for label in trainSet[0][0].keys():
			

			if label not in labelUsedList:

				gain = self.getGain(trainSet, label)

				if gain > maxGain:
					maxGain = gain
					maxGainLable = label

		if maxGainLable == "":
			
			self.finishBranch(labelList, trainSet)
			
			return

		else:

			labelList.append(maxGainLable)

			for item in self.featureValueDict[maxGainLable]:
				currentTrainSet = list(filter(lambda x:x[0][maxGainLable] == item, trainSet))

				self.nodeSelectionRecursion(currentTrainSet, labelList + [item], labelUsedList + [maxGainLable])


	def finishBranch(self, labelList, trainSet):

		tmpList = []

		# 生成一个分支列表
		iCount = 0
		for elem in labelList:

			if iCount == 0:
				key = elem
				iCount += 1

			else:
				value = elem
				tmpList.append((key,value))
				iCount = 0

		# 选择当前集合中，最普遍的结果作为最终结果
		tmpDict = {}
		for elem in trainSet:
			tmpDict[elem[1]] = 1 if elem[1] not in tmpDict.keys() else tmpDict[elem[1]] + 1
		tmpList.append(sorted(tmpDict.items(), key=lambda x:x[1], reverse=True)[0][0])

		self.treeList.append(tmpList)


	# get a dict like this
	# {"outlook":set(sunny, overcast, rain), }
	def getFeatureValueDict(self,trainSet):
		featureValueDict = {}
		for tupleItem in trainSet:

			for key, value in tupleItem[0].items():
				if key not in featureValueDict.keys():
					featureValueDict[key] = set()
				featureValueDict[key].add(value)
		return featureValueDict


	def classify(self, featureDict):
		
		for branch in self.treeList:

			for i in range(len(branch)):

				# 如果完全符合某个分支，则返回该分支的最后一项，即分类结果
				if i == len(branch) - 1:
					return branch[i]

				# 如果某个某个属性与分支不符，则换下一条分支
				elif featureDict.get(branch[i][0], None) != branch[i][1]:
					break
				
		# 如果遍历所有树分支都不能匹配待分类特征字典，则返回错误
		return "Error"


	# 熵度量样例的均一性
	def getEntropy(self, trainingSamplesList):

		tagDict = {}
		totalCount = 0

		for tupleItem in trainingSamplesList:

			tagDict[tupleItem[1]] = 1 if tupleItem[1] not in tagDict.keys() else tagDict[tupleItem[1]] + 1
			totalCount += 1

		# Entropy(S) = - p1 * log2(p1) - p2 * log2(p2)
		EntropyOfSystem = 0.0
		for key, value in tagDict.items():
			probability = tagDict[key] / totalCount
			EntropyOfSystem += - probability * log2(probability)

		return EntropyOfSystem


	# 信息增益度量期望的熵降低
	def getGain(self, trainingSamplesList, feature):

		# Gain(S,A) = Entropy(S) - sum(v属于Values(A))((Sv/S) * Entropy(Sv))

		gainDict = {}
		totalCount = 0
		for tupleItem in trainingSamplesList:
			gainDict[tupleItem[0][feature]] = 1 if tupleItem[0][feature] not in gainDict.keys() else gainDict[tupleItem[0][feature]] + 1
			totalCount += 1

		EntropyAfterUsingFeature = 0.0
		for key, value in gainDict.items():

			currentFeatureSampleList = list(filter(lambda x: x[0][feature] == key, trainingSamplesList))
			EntropyAfterUsingFeature += (value / totalCount) * self.getEntropy(currentFeatureSampleList)

		gain = self.getEntropy(trainingSamplesList) - EntropyAfterUsingFeature
		return gain



if __name__ == '__main__':
	

	# 网上找的例子，结果与网上手算的结果一致
	"""
	-----------------------------
	武器 | 子弹数量 | 血 | 行为
	-----------------------------
	机枪 | 多 | 少 | 战斗
	机枪 | 少 | 多 | 逃跑
	小刀 | 少 | 多 | 战斗
	小刀 | 少 | 少 | 逃跑
	"""
	classifyObject = DT()
	classifyObject.train(
		[
			({"武器":"机枪", "子弹数量":"多", "血":"少", }, "战斗"),
			({"武器":"机枪", "子弹数量":"少", "血":"多", }, "逃跑"),
			({"武器":"小刀", "子弹数量":"少", "血":"多", }, "战斗"),
			({"武器":"小刀", "子弹数量":"少", "血":"少", }, "逃跑"),
		]
		)
	res = classifyObject.classify({"武器":"机枪", "子弹数量":"多", "血":"少", })
	print(res)


	# 让我们把朴素贝叶斯中遇到的问题拿来看看，预测一下打喷嚏的建筑工人。
	classifyObject = DT()
	classifyObject.train(
		[
			({"症状":"打喷嚏", "职业":"护士"}, "感冒"),
			({"症状":"打喷嚏", "职业":"农夫"}, "过敏"),
			({"症状":"头痛", "职业":"建筑工人"}, "脑震荡"),
			({"症状":"头痛", "职业":"建筑工人"}, "感冒"),
			({"症状":"打喷嚏", "职业":"教师"}, "感冒"),
			({"症状":"头痛", "职业":"教师"}, "脑震荡"),
		]
		)
	res = classifyObject.classify({"症状":"头痛", "职业":"建筑工人"})
	print(res)

