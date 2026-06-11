import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib
# 解决中文乱码
matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 1. 加载鸢尾花数据集（4维特征）
iris = load_iris()
X = iris.data
y = iris.target
# 2. PCA降维：从4维降到2维
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
# 3. 可视化降维结果
plt.figure(figsize=(8, 5))
for target, color in zip(range(3), ['r', 'g', 'b']):
    plt.scatter(X_pca[y == target, 0], X_pca[y == target, 1],
                c=color, label=iris.target_names[target], alpha=0.7)
plt.title('PCA降维（鸢尾花数据集 4维→2维）')
plt.xlabel(f'主成分1（方差解释率: {pca.explained_variance_ratio_[0]:.2f}）')
plt.ylabel(f'主成分2（方差解释率: {pca.explained_variance_ratio_[1]:.2f}）')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
print(f"总方差解释率: {np.sum(pca.explained_variance_ratio_):.2f}")
print(f"主成分系数:\n{pca.components_}")