# feature-selection-method
一些常用的特征选择方法

# 常用的特征选择方法
- mRMR特征选择方法 (最大相关最小冗余方法)
- Removing features with low variance (移除低方差特征方法)
- Univariate feature selection (单变量特征选择方法)
- Locally linear embedding (局部线性嵌入方法)
- PCA (主成分分析法)
- kPCA (核主成分分析法)
- factor analysis (因子分析)
- ICA (独立成分分析方法)

# 数据要求
## 输入
- 第一列为: Y数据
- 第二列为: X数据

# 代码
```python 
# 导入特征选择库
import Feature_Selection as FS
import pandas as pd 

# 读取数据
data = pd.read_excel("数据文件.xlsx")


# mRMR 特征选择方法
feature_length = 10
selected_data = FS.mRMR_feature_selection(data,feature_length)

# Removing features with low variance
variance_threshold = 0.9
selected_data = FS.removing_features_with_low_variance(data,variance_threshold)

# Univariate feature selection
feature_length = 10
selected_data = FS.univariate_feature_selection(data,feature_length) 

# Locally linear embedding
feature_length = 10
selected_data = FS.lle_feature_reduction(data,feature_length)

# PCA
feature_length = 10
selected_data = FS.PCA_feature_selection(data,feature_length)

# Factor Analysis
feature_length = 10
selected_data = FS.fa_feature_selection(data,feature_length)

# ICA
selected_data = FS.ICA_feature_selection(data,feature_length)
```

