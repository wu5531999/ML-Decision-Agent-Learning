import gymnasium as gym
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


EPISODES = 200
TRAIN_EPISODES = 100  # 用于收集数据训练模型的回合数
TEST_EPISODES = 10  # 用于测试联动系统的回合数


env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n


print("📊 正在收集训练数据...")
X = []  # 状态特征：[位置, 速度, 杆角度, 角速度]
y = []  # 标签：下一个状态的杆角度（用于回归预测）

for e in range(TRAIN_EPISODES):
    state, _ = env.reset()
    for t in range(200):
        # 随机动作收集数据
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)

        # 收集当前状态作为特征，下一个状态的杆角度作为标签
        X.append(state)
        y.append(next_state[2])  # 杆角度是第三个状态变量

        state = next_state
        if terminated or truncated:
            break

# 转换为numpy数组
X = np.array(X)
y = np.array(y)


print("🤖 正在训练机器学习模型...")


# 把杆角度离散化，转为分类问题（方便用逻辑回归）
# 角度范围：-0.2095 到 0.2095，分成3类：左倾、垂直、右倾
def discretize_angle(angle):
    if angle < -0.05:
        return 0  # 左倾
    elif angle > 0.05:
        return 2  # 右倾
    else:
        return 1  # 垂直


y_discrete = np.array([discretize_angle(angle) for angle in y])

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_discrete, test_size=0.2, random_state=42)

# 训练逻辑回归模型
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 评估模型准确率
accuracy = model.score(X_test, y_test)
print(f"✅ 模型训练完成！测试集准确率: {accuracy:.2f}")



def make_decision(state):
    """根据模型预测的杆角度，生成最优动作"""
    # 用模型预测当前状态下，下一个状态的杆角度类别
    pred = model.predict(state.reshape(1, -1))[0]

    if pred == 0:  # 预测杆会向左倒，向右推
        return 1
    elif pred == 2:  # 预测杆会向右倒，向左推
        return 0
    else:  # 预测杆保持垂直，随机微调（保持探索性）
        return env.action_space.sample()



print("\n🚀 开始测试「机器学习+决策模型」联动系统...")
total_scores = []

for e in range(TEST_EPISODES):
    state, _ = env.reset()
    score = 0
    for t in range(200):
        # 调用决策模型生成动作
        action = make_decision(state)
        next_state, reward, terminated, truncated, _ = env.step(action)

        state = next_state
        score += reward

        if terminated or truncated:
            print(f"测试回合 {e + 1}/{TEST_EPISODES} | 得分: {score}")
            total_scores.append(score)
            break

# 输出测试结果
avg_score = np.mean(total_scores)
print(f"\n📈 联动系统测试完成！")
print(f"平均得分: {avg_score:.2f}（满分200）")
print("说明：得分越高，联动系统效果越好！")

env.close()


