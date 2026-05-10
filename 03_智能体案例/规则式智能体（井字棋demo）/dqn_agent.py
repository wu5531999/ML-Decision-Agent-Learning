import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_dim=9, action_dim=9, hidden_dim=64,
                 lr=1e-3, gamma=0.99, epsilon=0.1, buffer_capacity=10000, batch_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size

        self.policy_net = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def select_action(self, state, legal_actions):
        if np.random.rand() < self.epsilon:
            return np.random.choice(legal_actions)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state)
            # 只在合法动作中选择Q值最大的
            q_values = q_values.cpu().numpy().flatten()
            q_values[~np.isin(np.arange(self.action_dim), legal_actions)] = -np.inf
            return np.argmax(q_values)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None  # 改为返回None，而非不返回

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # 当前Q值
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # 下一个状态的最大Q值（加detach防止梯度回传）
        next_q = self.target_net(next_states).max(1)[0].detach()
        # 目标Q值（裁剪奖励，避免过大）
        target_q = rewards.clamp(-1, 1) + (1 - dones) * self.gamma * next_q.clamp(-1, 1)

        # 计算损失
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
    
        # 梯度裁剪（核心：防止梯度爆炸导致nan）
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
    
        self.optimizer.step()

        # 返回loss值（确保是浮点数）
        loss_val = loss.item()
        return loss_val if not np.isnan(loss_val) else None
 
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(torch.load(path))

