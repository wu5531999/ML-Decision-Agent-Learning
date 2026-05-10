import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib
# 解决中文乱码问题
matplotlib.use('TkAgg') 
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 优先用黑体/微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
# 1. 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target
# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 3. 训练决策树模型
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)
# 4. 预测并计算准确率
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
# 5. 可视化决策树
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=iris.feature_names, class_names=iris.target_names,
          filled=True, rounded=True, fontsize=10)
plt.title('CART决策树（鸢尾花数据集）')
plt.show()
# 6. 输出结果
print(f"测试集准确率: {acc:.2f}")
print("\n特征重要性:")
for name, importance in zip(iris.feature_names, model.feature_importances_):
    print(f"{name}: {importance:.4f}")