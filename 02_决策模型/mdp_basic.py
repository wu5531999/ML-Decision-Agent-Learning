import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# 解决中文乱码问题
matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 优先用黑体/微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
# 1. 定义4x4网格世界MDP
# 状态：0-15（共16个状态），状态15是终点（奖励+1），状态5、7、11、12是障碍（奖励-1）
# 动作：0=上, 1=下, 2=左, 3=右
n_states = 16
n_actions = 4
gamma = 0.9  # 折扣因子
theta = 1e-6  # 收敛阈值
# 奖励函数
R = np.zeros(n_states)
R[15] = 1.0  # 终点奖励
R[[5, 7, 11, 12]] = -1.0  # 障碍惩罚
# 状态转移函数
def transition(s, a):
    row, col = s // 4, s % 4
    if a == 0: row = max(row-1, 0)  # 上
    elif a == 1: row = min(row+1, 3)  # 下
    elif a == 2: col = max(col-1, 0)  # 左
    elif a == 3: col = min(col+1, 3)  # 右
    return row * 4 + col
# 2. 价值迭代算法
V = np.zeros(n_states)
while True:
    delta = 0
    for s in range(n_states):
        if s == 15: continue  # 终点状态价值为0
        v = V[s]
        # 计算所有动作的价值
        q_values = np.zeros(n_actions)
        for a in range(n_actions):
            s_next = transition(s, a)
            q_values[a] = R[s] + gamma * V[s_next]
        V[s] = np.max(q_values)
        delta = max(delta, abs(v - V[s]))
    if delta < theta:
        break
# 3. 提取最优策略
policy = np.zeros(n_states, dtype=int)
for s in range(n_states):
    if s == 15: continue
    q_values = np.zeros(n_actions)
    for a in range(n_actions):
        s_next = transition(s, a)
        q_values[a] = R[s] + gamma * V[s_next]
    policy[s] = np.argmax(q_values)
# 4. 可视化结果
action_names = ['↑', '↓', '←', '→']
policy_grid = np.array([action_names[a] for a in policy]).reshape(4, 4)
policy_grid[3, 3] = '★'  # 终点
policy_grid[[1,1,2,3], [1,3,3,0]] = '■'  # 障碍
value_grid = V.reshape(4, 4)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(value_grid, cmap='viridis')
plt.title('状态价值函数')
for i in range(4):
    for j in range(4):
        plt.text(j, i, f'{value_grid[i,j]:.2f}', ha='center', va='center', color='white')
plt.xticks([])
plt.yticks([])
plt.subplot(1, 2, 2)
plt.imshow(np.zeros((4,4)), cmap='gray')
plt.title('最优策略')
for i in range(4):
    for j in range(4):
        plt.text(j, i, policy_grid[i,j], ha='center', va='center', color='white', fontsize=20)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()
print("价值迭代完成！")
print("最优策略网格：")
print(policy_grid)