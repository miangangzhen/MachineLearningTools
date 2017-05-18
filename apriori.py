#! -*-coding=utf-8-*-
"""
经典的关联规则挖掘算法包括Apriori算法和FP-growth算法
"""
import itertools

def getC1L1(data, minSupport):

	C1 = {}
	labelSet = set()
	for row in data:
		for item in row:
			if item not in labelSet:
				labelSet.add(item)
				C1[item] = 0
			C1[item] += 1

	for label in C1.keys():
		if C1[label] < minSupport:
			labelSet.remove(label)
	return labelSet


def hasSameSubitem(str1Set, str2Set):

	if len(str1Set) != len(str2Set):
		return False
	return len(str1Set - str2Set) == 1


def hasFreqSubitem(KplusOneSetStr, originSet):
	KplusOneSetList = KplusOneSetStr.split(",")
	isFreq = True
	for item in itertools.combinations(KplusOneSetList,len(KplusOneSetList) - 1):
		tmpList = list(item)
		tmpList.sort()
		tmpStr = ",".join(tmpList)
		if tmpStr not in originSet:
			isFreq = False
			break
	return isFreq


def getCandidateSet(originSet):
	"""由每个元素长度为K组成的频繁项集originSet, 构成每个元素长度为K+1的候选项集nextCandidateSet
	"""
	nextCandidateSet = set()
	# 自连接
	for itemI, itemJ in itertools.combinations(originSet,2):
		itemISet = set(itemI.split(","))
		itemJSet = set(itemJ.split(","))
		# 如果两个k项集itemI&itemJ可以自连接, 必须保证它们有k-1项是相同的
		if hasSameSubitem(itemISet, itemJSet):
			# 取并集
			tmpList = list(itemISet | itemJSet)
			tmpList.sort()
			tmpStr = ",".join(tmpList)
			nextCandidateSet.add(tmpStr)

	# 剪枝步（这个步骤是为了压缩nextCandidate的大小, 减少扫描数据样本的次数）
	# 如果K+1个元素构成频繁项集, 那么它的任意K个元素的子集也是频繁项集
	# 必须保证它的所有K个元素的子集都是频繁的
	tmpList = list(nextCandidateSet)
	for item in tmpList:
		if not hasFreqSubitem(item, originSet):
			nextCandidateSet.remove(item)
	return nextCandidateSet


def getSupport(item, data):

	support = 0
	item = set(item.split(","))
	for row in data:
		if item.issubset(set(row)):
			support += 1
	return support


def judge(originSet, minSupport, data):

	CandidateDict = {}
	# 剪枝步
	# 计算候选项集的支持度
	for item in originSet:
		CandidateDict[item] = getSupport(item, data)
	# 频繁项集的支持度要大于最小支持度
	for item in originSet:
		if CandidateDict[item] < minSupport:
			del CandidateDict[item]
	return set(CandidateDict.keys())


def apriori(data, minSupport=2):
	"""Main function of algorithm
	:param data: training data
	:param minSupport: number of min support
	
	Apriori算法属于关联分析, 是Agrawal在1993提出来的
	它的功能是找出频繁项集
	"""
	# 计算初始项集, 筛选得到频繁集
	CandidateSet = getC1L1(data, minSupport)
	# 不断进行自连接和剪枝, 直到得到最终的频繁集为止;终止条件是, 如果自连接得到的已经不再是频繁集
	# 那么取最后一次得到的频繁集作为结果
	while 1:
		resultSet = CandidateSet

		newCandidateSet = getCandidateSet(CandidateSet)

		CandidateSet = judge(newCandidateSet, minSupport, data)
		if len(CandidateSet) == 0:
			# print(resultSet)
			return resultSet
		# print(CandidateSet)


if __name__ == '__main__':

	# 最小支持度
	# 最小支持度是22%, 对于samples1, 那么每件商品至少要出现9*22%=2次才算频繁
	minSupport = 2
	
	# sample1
	samples = [
		["I1","I2","I5"],
		["I2","I4"],
		["I2","I3"],
		["I1","I2","I4"],
		["I1","I3"],
		["I2","I3"],
		["I1","I3"],
		["I1","I2","I3","I5"],
		["I1","I2","I3"]
	]
	result = apriori(samples, minSupport)
	print("最终频繁项集：")
	print(result)

	# sample2
	samples = [
		["A", "C", "D"],
		["B", "C", "E"],
		["A", "B", "C", "E"],
		["B", "E"]
	]
	result = apriori(samples, minSupport)
	print("最终频繁项集：")
	print(result)
	
	# sample3
	minSupport = 3
	samples = [
		["M", "O", "N", "K", "E", "Y"],
		["D", "O", "N", "K", "E", "Y"],
		["M", "A", "K", "E"],
		["M", "A", "C", "K", "Y"],
		["C", "O", "O", "K", "I", "E"],
	]
	result = apriori(samples, minSupport)
	print("最终频繁项集：")
	print(result)
	
	# sample4
	minSupport = 3
	samples = [
		['bread', 'milk'],
		['bread', 'diaper', 'beer', 'egg'],
		['milk', 'diaper', 'beer', 'cola'],
		['bread', 'milk', 'diaper', 'beer'],
		['bread', 'milk', 'diaper', 'cola'],
	]
	result = apriori(samples, minSupport)
	print("最终频繁项集：")
	print(result)

	# sample5
	samples = []
	with open("data.csv", "r") as f:
		for line in f.readlines():
			samples.append(line.strip('\n').split(','))
	minSupport = int(len(samples) * 20 / 100)
	result = apriori(samples, minSupport)
	print("最终频繁项集：")
	print(result)

	# sample6
	minSupport = 3
	samples = [
	["牛奶", "鸡蛋", "面包", "薯片"],
	["鸡蛋", "爆米花", "薯片", "啤酒"],
	["鸡蛋", "面包", "薯片"],
	["牛奶", "鸡蛋", "面包", "爆米花", "薯片", "啤酒"],
	["牛奶", "面包", "啤酒"],
	["鸡蛋", "面包", "啤酒"],
	["牛奶", "面包", "薯片"],
	["牛奶", "鸡蛋", "面包", "黄油", "薯片"],
	["牛奶", "鸡蛋", "黄油", "薯片"],
	]
	result = apriori(samples, minSupport)
	print("最终频繁项集：")
	print(result)