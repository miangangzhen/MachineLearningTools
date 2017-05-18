import numpy as np
  
# 1）计算已知类别数据集中的点与当前点之间的距离；

# 2）按照距离递增次序排序；

# 3）选取与当前点距离最小的k个点；

# 4）确定前k个点所在类别的出现频率；

# 5）返回前k个点出现频率最高的类别作为当前点的预测分类。
def KnnClassify(sample, dataSet, labels, K):

	difference = dataSet - sample
	difference = np.power(difference, 2)
	distance = difference.sum(1)
	distance = distance ** 0.5
	sortdiffidx = distance.argsort()

	vote = {}
	for i in range(K):
		label = labels[sortdiffidx[i]]
		vote[label] = vote.get(label, 0) + 1
	sortedvote = sorted(vote.items(),key = lambda x:x[1], reverse = True)
	return sortedvote[0][0]


if __name__ == '__main__':

	group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])  
	labels = ['A','A','B','B']  
	
	res = KnnClassify([0,0], group,labels,3)
	print(res)