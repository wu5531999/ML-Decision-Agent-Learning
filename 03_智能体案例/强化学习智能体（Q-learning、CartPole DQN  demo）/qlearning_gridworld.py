import numpy as np

# 网格世界环境
GRID_SIZE = 5
START = (0, 0)
GOAL = (4, 4)
OBSTACLES = [(1, 1), (3, 3)]
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右

# Q-learning 参数
alpha = 0.1
gamma = 0.9
epsilon = 0.1
episodes = 500

# Q 表初始化
Q = np.zeros((GRID_SIZE, GRID_SIZE, 4))

# 单步移动
def step(state, action):
    i, j = state
    di, dj = ACTIONS[action]
    ni, nj = i + di, j + dj
    if 0 <= ni < GRID_SIZE and 0 <= nj < GRID_SIZE:
        if (ni, nj) not in OBSTACLES:
            return (ni, nj)
    return state

# 训练
for _ in range(episodes):
    s = START
    while s != GOAL:
        # 动作选择
        if np.random.rand() < epsilon:
            a = np.random.choice(4)
        else:
            a = np.argmax(Q[s[0], s[1]])
        s_next = step(s, a)
        # 奖励函数
        if s_next == GOAL:
            r = 10
        elif s_next in OBSTACLES:
            r = -1
        else:
            r = -0.1
        # Q 更新
        Q[s[0], s[1], a] = (1 - alpha) * Q[s[0], s[1], a] + alpha * (r + gamma * np.max(Q[s_next[0], s_next[1]]))
        s = s_next

# 最优路径
path = []
s = START
while s != GOAL and len(path) < 30:
    path.append(s)
    a = np.argmax(Q[s[0], s[1]])
    s = step(s, a)
path.append(GOAL)

print("最优路径：")
print(path)
