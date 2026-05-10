import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib
# 解决中文乱码
matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 1. 生成模拟聚类数据
np.random.seed(42)
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)
# 2. 训练K-Means模型
kmeans = KMeans(n_clusters=4, random_state=42)
y_pred = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_
# 3. 可视化聚类结果
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis', alpha=0.7)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.9, marker='*', label='聚类中心')
plt.title('K-Means聚类 Demo')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
print(f"聚类中心坐标:\n{centers}")
print(f"迭代次数: {kmeans.n_iter_}")