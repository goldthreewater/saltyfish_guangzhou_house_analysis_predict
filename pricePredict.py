# 导入数据统计模块
from math import sqrt

import pandas
# 导入回归函数
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import numpy as np

from sklearn.svm import LinearSVR
# 读取csv数据文件
data = pandas.read_csv('爬取结果/baiyun.csv', encoding='gb18030', header=None)
data.columns = ['名称', '小区名', '位置', '户型', '建筑面积', '朝向', '装修', '楼层', '年份', '楼况', '关注人数', '发布时间', '总价', '单价', '标签']
data['单价'] = data['单价'].str.replace(",", "")
data['单价'] = data['单价'].str.replace("元/平", "")
# 将索引列删除
#del data['Unnamed: 0']
# 删除data数据中的所有空值
data.dropna(axis=0, how='any', inplace=True)
# 将总价“万”去掉
#data['总价'] = data['总价'].map(lambda z: z.replace('万', ''))
# 将房子总价转换为浮点类型
data['总价'] = data['总价'].astype(float)
# 将建筑面价“平米”去掉
#data['建筑面积'] = data['建筑面积'].map(lambda p: p.replace('平米', ''))
# 将建筑面积转换为浮点类型
data['建筑面积'] = data['建筑面积'].astype(float)
# 拷贝数据
data_copy = data.copy()
#显示‘户型’、‘建筑面积’的头部信息，前五行
print('显示‘户型’、‘建筑面积’的头部信息，前五行:\n',data_copy[['户型', '建筑面积']].head())
#处理户型字段
data_copy[['室', '厅']] = data_copy['户型'].str.extract('(\d+)室(\d+)厅')
# 将房子室转换为浮点类型
data_copy['室'] = data_copy['室'].astype(float)
# 将房子厅转换为浮点类型
data_copy['厅'] = data_copy['厅'].astype(float)
# 打印“室”、“厅”、“卫”数据
print('打印处理后的“室”、“厅”数据：\n',data_copy[['室','厅']].head())
#将没有用的字段删除
del data_copy['名称']
del data_copy['小区名']
del data_copy['位置']
del data_copy['户型']
#del data_copy['朝向']
#del data_copy['楼层']
del data_copy['装修']
del data_copy['标签']
del data_copy['发布时间']
del data_copy['楼况']
del data_copy['单价']
del data_copy['年份']
# 删除data数据中的所有空值
data_copy.dropna(axis=0, how='any', inplace=True)
print('处理后的头部信息：\n',data_copy.head())
data_copy.dropna(axis=0, inplace=True)

from sklearn.preprocessing import LabelEncoder

# Create a label encoder object
le = LabelEncoder()
le_count = 0

# Iterate through the columns
for col in data_copy:
    if data_copy[col].dtype == 'object':
        # If 2 or fewer unique categories
        # if len(list(df[col].unique())) <= 2:
        print(col)
        if 1:
            # Train on the training data
            le.fit(data_copy[col])
            # Transform both training and testing data
            data_copy[col] = le.transform(data_copy[col])


#import pdb
#pdb.set_trace()
y = data_copy["总价"]
data_copy.drop(["总价"], axis=1, inplace=True)
x = data_copy.copy()
data_copy.to_csv("test1.csv")
y.to_csv("testy.csv")

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
reg = GradientBoostingRegressor(random_state=0)
reg.fit(x_train, y_train)
y_predict = reg.predict(x_test)
print("mean_absolute_error:", mean_absolute_error(y_test, y_predict))
print("mean_squared_error:", mean_squared_error(y_test, y_predict))
print("rmse:", sqrt(mean_squared_error(y_test, y_predict)))
scores = cross_val_score(reg,x,y,scoring='r2',cv=5)
print (scores)


regressor = RandomForestRegressor(n_estimators=100,random_state=0)
regressor.fit(x_train, y_train)
y_predict = regressor.predict(x_test)
print("mean_absolute_error:", mean_absolute_error(y_test, y_predict))
print("mean_squared_error:", mean_squared_error(y_test, y_predict))
print("rmse:", sqrt(mean_squared_error(y_test, y_predict)))
#scores = cross_val_score(regressor,x,y,scoring='mean_squared_error',cv=5).mean()
scores = cross_val_score(regressor,x,y,scoring='r2',cv=5)
print (scores)


plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
fig=plt.figure(figsize=(10,6))
feature_names=data_copy.columns.to_list()
feature_names = np.array(feature_names)
print("特征重要性："+str(regressor.feature_importances_))
sorted_idx = regressor.feature_importances_.argsort()
plt.barh(feature_names[sorted_idx], regressor.feature_importances_[sorted_idx])
plt.xlabel("Random Forest Feature Importance")
plt.savefig("importance.jpg")
plt.show()