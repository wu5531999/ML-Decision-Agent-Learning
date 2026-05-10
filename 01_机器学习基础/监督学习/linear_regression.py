import numpy as np
import matplotlib.pyplot as plt
# 解决matplotlib画图不显示问题
import matplotlib
# 解决中文乱码问题
matplotlib.use('TkAgg')  
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 优先用黑体/微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
# 1. 生成模拟数据：y = 2x + 5 + 噪声
np.random.seed(42)  # 固定随机种子，结果可复现
x = np.linspace(0, 10, 100)
y = 2 * x + 5 + np.random.randn(100) * 0.8
# 2. 最小二乘法求解线性回归
X = np.vstack([x, np.ones(len(x))]).T  # 构造增广矩阵
w, b = np.linalg.lstsq(X, y, rcond=None)[0]
# 3. 预测
y_pred = w * x + b
# 4. 可视化结果
plt.figure(figsize=(8, 5))
plt.scatter(x, y, alpha=0.7, label='原始数据')
plt.plot(x, y_pred, 'r-', lw=2, label=f'拟合直线: y={w:.2f}x+{b:.2f}')
plt.title('线性回归 Demo')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
# 5. 输出结果
print(f"拟合参数：斜率w={w:.2f}, 截距b={b:.2f}")
print(f"均方误差MSE: {np.mean((y_pred - y)**2):.4f}")