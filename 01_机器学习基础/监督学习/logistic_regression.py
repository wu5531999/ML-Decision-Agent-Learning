import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib
# 解决中文乱码问题
matplotlib.use('TkAgg') 
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 优先用黑体/微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
# 1. 加载鸢尾花数据集（只取前两类，做二分类）
iris = load_iris()
X = iris.data[:100, :2]  # 只取前两个特征，方便可视化
y = iris.target[:100]
# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 3. 训练逻辑回归模型
model = LogisticRegression()
model.fit(X_train, y_train)
# 4. 预测并计算准确率
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
# 5. 可视化分类边界
x1_min, x1_max = X[:, 0].min()-0.5, X[:, 0].max()+0.5
x2_min, x2_max = X[:, 1].min()-0.5, X[:, 1].max()+0.5
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),
                       np.arange(x2_min, x2_max, 0.02))
Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()])
Z = Z.reshape(xx1.shape)
plt.figure(figsize=(8, 5))
plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
plt.title(f'逻辑回归二分类（鸢尾花数据集）\n准确率: {acc:.2f}')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()
print(f"测试集准确率: {acc:.2f}")