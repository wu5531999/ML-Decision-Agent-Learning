import gymnasium as gym
import numpy as np
import random
from collections import deque

EPISODES = 300          # 训练总回合数
BATCH_SIZE = 64         # 增大batch size，降低梯度噪声
MEMORY_SIZE = 5000      # 增大经验池容量
GAMMA = 0.99            # 提高折扣因子，更看重长期奖励
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.99    # 加快衰减，让epsilon更快降到0.01
LR = 0.005              # 适当提高学习率，加快收敛

env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

class LinearDQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.weights = np.random.randn(state_size, action_size) * 0.01
    def predict(self, state):
        return np.dot(state, self.weights)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.model = LinearDQN(state_size, action_size)
        self.lr = LR
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        states = np.vstack([s[0] for s in minibatch])
        actions = np.array([s[1] for s in minibatch])
        rewards = np.array([s[2] for s in minibatch])
        next_states = np.vstack([s[3] for s in minibatch])
        dones = np.array([s[4] for s in minibatch])
        
        targets = rewards.copy()
        non_final_mask = ~dones
        next_q_values = self.model.predict(next_states[non_final_mask])
        targets[non_final_mask] += self.gamma * np.amax(next_q_values, axis=1)
        
        current_q_values = self.model.predict(states)
        current_q_values[np.arange(batch_size), actions] = targets
        
        pred_q_values = self.model.predict(states)
        gradient = np.dot(states.T, (current_q_values - pred_q_values)) / batch_size
        self.model.weights += self.lr * gradient

agent = DQNAgent(state_size, action_size)
print("开始训练...")
for e in range(EPISODES):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    total_score = 0
    for time in range(200):
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # 修正：先计算额外奖励，再reshape，或者直接用一维索引
        reward = reward if not done else -10
        if not done:
            # 直接用一维索引访问原始 next_state
            reward += 1 - abs(next_state[2]) * 2
        
        # 现在再 reshape 成二维
        next_state = np.reshape(next_state, [1, state_size])
        
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_score += 1
        
        if len(agent.memory) > BATCH_SIZE:
            agent.replay(BATCH_SIZE)
        
        if done:
            print(f"Episode: {e+1}/{EPISODES}, Score: {total_score}, Epsilon: {agent.epsilon:.2f}")
            break
    
    # 按回合衰减探索率
    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay

print("\n训练完成，开始测试...")
env = gym.make("CartPole-v1")
state, _ = env.reset()
state = np.reshape(state, [1, state_size])
score = 0
agent.epsilon = 0  # 测试时关闭探索
for time in range(200):
    action = agent.act(state)
    next_state, reward, terminated, truncated, _ = env.step(action)
    state = np.reshape(next_state, [1, state_size])
    score += reward
    if terminated or truncated:
        break
print(f"测试结果：杆子坚持了 {score} 步（满分200）")
print("如果分数接近200，说明智能体已经学会了平衡策略！")
env.close()
