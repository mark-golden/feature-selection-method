
"""
程序作者: GRZ
程序时间: 2021-04-14
程序目的: 特征选择的调用
"""

# Import Library
import Feature_Selection as FS
import pandas as pd 

def main():
	# 原始数据
	input_file = 'Sample_data.xlsx'
	data = pd.read_excel(input_file)

	# 输入要提取的特征数
	feature_length = 5
	# X与Y的类型: continuous/discrete
	X_type = 'continuous'
	Y_type = 'discrete'
	
	# 特征选择
	selected_data = FS.mRMR_feature_selection(data,feature_length,X_type,Y_type)

	print(selected_data)
	


if __name__ == '__main__':
	main()


