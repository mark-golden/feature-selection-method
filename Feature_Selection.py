

import pandas as pd
import numpy as np
from sklearn import metrics
# 导入mutual_info_regression库
from sklearn.feature_selection import mutual_info_regression

import warnings
warnings.filterwarnings("ignore")

"""
	类: feature_1_select
	功能: 选择第一个特征
	输入: dataset-数据集
	变量含义: 
		self.NMI_dict: { '1': ['1', X1向量, MI1], '2': ['2', X2向量, MI2], … }
		self.NMI_dict_revise: 对 self.NMI_dict 排序并去掉第1个特征后的新字典
		feature_1: 选择的第1个特征 ['1', X1向量, MI1]
"""
class feature_1_select():
	def __init__(self,dataset):

		# 数据列数
		col_number = dataset.shape[1]

		# Y数据
		output_data = dataset.iloc[:,0:1]
		output_label = np.array(output_data).ravel() # 转换为一维数组
		# 检测output_label的数值范围是离散值还是连续值
		unique_values_y = np.unique(output_label)
		if len(unique_values_y) <= 15:
			Y_type = 'discrete'
		else:
			Y_type = 'continuous'

		# 互信息字典
		self.NMI_dict = {}
		for i in range(1,col_number):
			## X数据
			### 序号
			feature_name = str(i)
			### 提取X数据
			input_data = dataset.iloc[:,i:i+1]
			input_label = np.array(input_data).ravel() # 转换为一维数组
			### 检测input_label的数值范围是离散值还是连续值
			unique_values_x = np.unique(input_label)
			if len(unique_values_x) <= 15:
				X_type = 'discrete'
			else:
				X_type = 'continuous'

			## 计算互信息 
			### 适合于X与Y均为离散型变量
			if X_type == 'discrete' and Y_type == 'discrete':
				result_NMI = metrics.normalized_mutual_info_score(input_label, output_label)
			### 适合于X为连续型变量，Y为离散型变量
			elif X_type == 'continuous' and Y_type == 'discrete':
				# result_NMI = metrics.normalized_mutual_info_score(np.digitize(input_label, bins=10), output_label)
				result_NMI = metrics.normalized_mutual_info_score(input_label, output_label)
			### 适合于X与Y均为连续型变量
			elif X_type == 'continuous' and Y_type == 'continuous':
				result_NMI = mutual_info_regression(input_data, output_label)[0]

			self.NMI_dict[feature_name] = [str(i),input_label,result_NMI]

	def bubblesort(self):

		# 字典名
		key_list = list(self.NMI_dict.keys())

		# 排序
		n = len(key_list)
		for i in range(n):
			for j in range(0, n-i-1):
				data_1 = self.NMI_dict[key_list[j]][2]
				data_2 = self.NMI_dict[key_list[j+1]][2]

				if data_1 < data_2:
					self.NMI_dict[key_list[j]],self.NMI_dict[key_list[j+1]] = self.NMI_dict[key_list[j+1]], self.NMI_dict[key_list[j]]  

		# 修改字典
		NMI_dict_revise = {}
		for i in range(len(key_list)):
			NMI_dict_revise[self.NMI_dict[key_list[i]][0]] = self.NMI_dict[key_list[i]]

		# 修改字典key
		NMI_revise_list = list(NMI_dict_revise.keys())
		feature_1 = NMI_dict_revise[NMI_revise_list[0]]

		# 删除已选特征
		del NMI_dict_revise[NMI_revise_list[0]]
		
		return [NMI_dict_revise,feature_1]

"""
	类: 选择第2个及之后的特征
	变量含义:
		 feature_2: 当前轮次选出的特征（mRMR值最高）: ['3', X3_array, MI(X3, Y)]
		 NMI_dict_revise_2: 剩余未被选择的特征（下一轮的候选池）: {'4': ['4', X4_array, MI(X4, Y)], '5': ['5', X5_array, MI(X5, Y)]}

"""
class feature_i_select():
	def __init__(self,NMI_dict_revise,selected_feature):

		# 尚未选择的特征
		self.NMI_dict_revise = NMI_dict_revise
		selected_feature_length = len(selected_feature)
		NMI_revise_list = list(self.NMI_dict_revise.keys())

		# 计算已选特征和要添加的特征的MI
		for keys_value in NMI_revise_list:

			## 输入数据
			input_data = self.NMI_dict_revise[keys_value][1]
			### 检测input_data的数值范围是离散值还是连续值
			unique_values_x = np.unique(input_data)
			if len(unique_values_x) <= 15:
				X_type = 'discrete'
			else:
				X_type = 'continuous'
			
			## 最小冗余
			min_red = 0
			for feature_n in  selected_feature:
				feature_1_data = feature_n[1]
				### 检测feature_1_data的数值范围是离散值还是连续值
				unique_values_y = np.unique(feature_1_data)
				if len(unique_values_y) <= 15:
					Y_type = 'discrete'
				else:
					Y_type = 'continuous'
				
				### 适合于X与Y均为离散型变量
				if X_type == 'discrete' and Y_type == 'discrete':
					result_NMI = metrics.normalized_mutual_info_score(input_data, feature_1_data)
				### 适合于X为连续型变量，Y为离散型变量
				elif X_type == 'continuous' and Y_type == 'discrete':
					# result_NMI = metrics.normalized_mutual_info_score(np.digitize(input_data, bins=10), feature_1_data)
					result_NMI = metrics.normalized_mutual_info_score(input_data, feature_1_data)
				### 适合于X与Y均为连续型变量
				elif X_type == 'continuous' and Y_type == 'continuous':
					result_NMI = mutual_info_regression(input_data.reshape(-1, 1), feature_1_data.ravel())[0]

				try:
					min_red += result_NMI
				except:
					result_NMI = 0
					min_red += result_NMI

			## 平均冗余值
			mean_min_red = min_red / selected_feature_length
			
			## mRMR值
			mRMR = self.NMI_dict_revise[keys_value][2] - mean_min_red

			self.NMI_dict_revise[keys_value].append(mRMR)

	def bubblesort_2(self):

		# 字典名
		key_list = list(self.NMI_dict_revise.keys())

		# 排序
		n = len(key_list)
		for i in range(n):
			for j in range(0, n-i-1):
				data_1 = self.NMI_dict_revise[key_list[j]][3]
				data_2 = self.NMI_dict_revise[key_list[j+1]][3]

				if data_1 < data_2:
					self.NMI_dict_revise[key_list[j]],self.NMI_dict_revise[key_list[j+1]] = self.NMI_dict_revise[key_list[j+1]], self.NMI_dict_revise[key_list[j]]  

		# 修改字典
		NMI_dict_revise_2 = {}
		for i in range(len(key_list)):
			NMI_dict_revise_2[self.NMI_dict_revise[key_list[i]][0]] = self.NMI_dict_revise[key_list[i]][0:3]

		# 修改字典key
		NMI_revise_list_2 = list(NMI_dict_revise_2.keys())
		feature_2 = NMI_dict_revise_2[NMI_revise_list_2[0]][0:3]

		# 删除已选特征
		del NMI_dict_revise_2[NMI_revise_list_2[0]]
		
		return [NMI_dict_revise_2,feature_2]

"""
	类: 返回序列序号
"""
class output_data:
	def __init__(self):
		self.feature_series = []

	def get_selected_series(self,selected_feature):
		for item in selected_feature:
			feature_num = item[0]
			self.feature_series.append(feature_num)

	def print_data(self,dataset):
		# 输出数据
		self.final_data = dataset.iloc[:,0:1]

		# 提取数据
		col_number = dataset.shape[1]
		for i in range(1,col_number):
			feature_name = str(i)
			for j in self.feature_series:
				if feature_name == j:
					selected_data = dataset.iloc[:,i:i+1]
					self.final_data = pd.concat([self.final_data,selected_data],axis = 1)
		
		return self.final_data

"""
	函数: data_normalization
	功能: 数据归一化
	输入: data-数据集
	输出: data_normal-归一化后的数据集
"""	
def data_normalization(data):

	# 提取Y数据
	data_normal = data.iloc[:, 0:1]

	# 归一化x数据
	## 提取列数
	ncol = data.shape[1]
	## 提取列名
	headline_list = data.columns.tolist()
	## 循环归一化
	for col_number in range(1,ncol):
		# 提取列名
		headline = headline_list[col_number]
		# 归一化
		data_1 = data.iloc[:,col_number:col_number + 1]
		deviation_value = data_1.std()
		mean_value = data_1.mean()
		z_score_value = (data_1 - mean_value) / deviation_value
		# 添加到归一化数据中
		data_normal[headline] = z_score_value

	return data_normal


def data_normalization_0_1(data):
	# 提取Y数据
	data_normal = data.iloc[:, 0:1]

	# 归一化x数据
	## 提取列数
	ncol = data.shape[1]
	## 提取列名
	headline_list = data.columns.tolist()
	## 循环归一化
	for col_number in range(1,ncol):
		# 提取列名
		headline = headline_list[col_number]
		# 归一化
		data_1 = data.iloc[:,col_number:col_number + 1]
		min_value = data_1.min()
		max_value = data_1.max()
		z_score_value = (data_1 - min_value) / (max_value - min_value)
		# 添加到归一化数据中
		data_normal[headline] = z_score_value

	return data_normal


"""
	函数: mRMR_feature_selection
	功能: mRMR特征选择主函数
	输入: data-数据集
		  feature_length-要选择的特征数
	输出: selected_data-选择后的数据集
"""
def mRMR_feature_selection(data,feature_length):

	# 数据归一化
	data_normal = data_normalization(data)

	# 空缺值处理
	data_normal = data_normal.fillna(0)

	# 选择特征
	## 初始化
	selected_feature = []

	## 选出第一个特征
	[NMI_dict_revise,feature_1] = feature_1_select(data_normal).bubblesort()
	selected_feature.append(feature_1)

	## 选出第二及之后的特征
	feature_new_length = feature_length-1 # 还需要再选择特征的长度
	for i in range(feature_new_length):
		[NMI_dict_revise,feature] = feature_i_select(NMI_dict_revise,selected_feature).bubblesort_2()
		selected_feature.append(feature)

	## 返回序列序号
	data_output = output_data()
	data_output.get_selected_series(selected_feature)
	selected_data = data_output.print_data(data_normal)

	return selected_data


"""
	函数: Removing features with low variance
"""
def removing_features_with_low_variance(data,threshold_value):
	# 导入库
	from sklearn.feature_selection import VarianceThreshold

	# 选取非ST的数据
	data_feature = data.iloc[:,1:]

	# 构建模型
	sel = VarianceThreshold(threshold = threshold_value)
	
	# 训练模型
	selected_feature = sel.fit_transform(data_feature)

	# 将数据转换为pandas数据
	selected_feature_df = pd.DataFrame(selected_feature)

	# 拼接数据
	Y_data = data.iloc[:,0:1]
	concat_data = pd.concat([Y_data,selected_feature_df],axis=1)

	# 数据归一化
	data_normal = data_normalization(concat_data)

	return data_normal

"""
	函数: Univariate feature selection
"""
def univariate_feature_selection(data,feature_length):
	# 导入库
	from sklearn.feature_selection import SelectKBest
	from sklearn.feature_selection import chi2

	# 数据归一化
	data_normal = data_normalization_0_1(data)

	# 提取数据
	y = data_normal.iloc[:,0:1]
	X = data_normal.iloc[:,1:]

	# 选择Kbest feature
	X_new = SelectKBest(chi2,k=feature_length).fit_transform(X,y)

	# 制作df数据
	selected_feature_df = pd.DataFrame(X_new)

	# 拼接数据
	Y_data = data.iloc[:,0:1]
	concat_data = pd.concat([Y_data,selected_feature_df],axis=1)

	return concat_data

"""
	函数: Locally linear embedding
"""
def lle_feature_reduction(data,n_components):
	# 导入库
	from sklearn.manifold import locally_linear_embedding

	# 提取特征数据
	X = data.iloc[:,1:]

	# 特征降维
	X_new,Componeent_value = locally_linear_embedding(X,n_neighbors=10,n_components=n_components)

	# 制作df数据
	selected_feature_df = pd.DataFrame(X_new)

	# 拼接数据
	Y_data = data.iloc[:,0:1]
	concat_data = pd.concat([Y_data,selected_feature_df],axis=1)

	return concat_data

"""
	函数: PCA
"""
def PCA_feature_selection(data,n_components):
	# 导入库
	from sklearn.decomposition import PCA 

	# 数据归一化
	data_normal = data_normalization(data)

	# 提取特征数据
	X = data_normal.iloc[:,1:]

	# 建模数据
	pca = PCA(n_components = n_components)
	X_new = pca.fit_transform(X)

	# 制作df数据
	selected_feature_df = pd.DataFrame(X_new)

	# 拼接数据
	Y_data = data.iloc[:,0:1]
	concat_data = pd.concat([Y_data,selected_feature_df],axis=1)

	return concat_data


"""
	函数: kPCA
"""
def kPCA_feature_selection(data,n_components):
	# 导入库
	from sklearn.decomposition import KernelPCA 

	# 数据归一化
	data_normal = data_normalization(data)

	# 提取特征数据
	X = data_normal.iloc[:,1:]

	# 建模数据
	kpca = KernelPCA(n_components = n_components,kernel='poly')
	X_new = kpca.fit_transform(X)

	# 制作df数据
	selected_feature_df = pd.DataFrame(X_new)

	# 拼接数据
	Y_data = data.iloc[:,0:1]
	concat_data = pd.concat([Y_data,selected_feature_df],axis=1)

	return concat_data

"""
	函数: factor analysis
"""
def fa_feature_selection(data,n_components):
	# 导入库
	from sklearn.decomposition import FactorAnalysis 

	# 数据归一化
	data_normal = data_normalization(data)

	# 提取特征数据
	X = data_normal.iloc[:,1:]

	# 建模数据
	fa = FactorAnalysis(n_components = n_components,random_state=0)
	X_new = fa.fit_transform(X)

	# 制作df数据
	selected_feature_df = pd.DataFrame(X_new)

	# 拼接数据
	Y_data = data.iloc[:,0:1]
	concat_data = pd.concat([Y_data,selected_feature_df],axis=1)

	return concat_data

"""
	函数: ICA
"""
def ICA_feature_selection(data,n_components):
	# 导入库
	from sklearn.decomposition import FastICA 

	# 数据归一化
	data_normal = data_normalization(data)

	# 提取特征数据
	X = data_normal.iloc[:,1:]

	# 建模数据
	ica = FastICA(n_components = n_components,random_state=0)
	X_new = ica.fit_transform(X)

	# 制作df数据
	selected_feature_df = pd.DataFrame(X_new)

	# 拼接数据
	Y_data = data.iloc[:,0:1]
	concat_data = pd.concat([Y_data,selected_feature_df],axis=1)

	return concat_data
