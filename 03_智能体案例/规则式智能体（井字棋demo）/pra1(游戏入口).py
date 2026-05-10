import os
import sys
import io
# 强制设置终端编码为UTF-8
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.system('chcp 65001 >nul')
# 重定向标准输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')
import numpy as np
import torch
from env import TicTacToe, play_episode, play_human_vs_agent
from dqn_agent import DQNAgent
import numpy as np

def train_agent(env, agent, episodes=2000, target_update=100):
    """训练函数"""
    rewards = []
    losses = []
    # 初始化epsilon衰减（探索率慢慢降低，从0.9到0.1）
    start_epsilon = 0.9
    end_epsilon = 0.1
    epsilon_decay = (start_epsilon - end_epsilon) / episodes
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        # 每局最多走9步（井字棋满格），防止死循环
        step_count = 0
        max_steps = 9
         # 衰减探索率
        agent.epsilon = max(end_epsilon, start_epsilon - episode * epsilon_decay)
        while not done and step_count < max_steps:
            step_count += 1
            # 选择动作
            legal_actions = env.get_legal_actions()
            action = agent.select_action(state, legal_actions)
            # 执行动作
            next_state, reward, done, info = env.step(action)
            # 修正奖励（避免数值过大）
            # 获胜+1，失败-1，平局0，每步存活+0.1（鼓励多走）
            if info["win"] == 1:
                reward = 1.0
            elif info["win"] == -1:
                reward = -1.0
            elif info["win"] == 0:
                reward = 0.0
            else:
                reward = 0.1  # 未结束时的小奖励
            # 存入经验池
            agent.replay_buffer.push(state, action, reward, next_state, done)
            # 更新模型（只有经验池足够时才更新）
            if len(agent.replay_buffer) >= agent.batch_size:
                loss = agent.update()
                if loss is not None and not np.isnan(loss):  # 过滤nan
                    losses.append(loss)
            state = next_state
            total_reward += reward
        rewards.append(total_reward)
        # 更新目标网络（固定频率）
        if episode % target_update == 0 and len(losses) > 0:
            agent.update_target_net()
        # 打印进度（只在有loss时计算平均值）
        if episode % 100 == 0:
            avg_reward = np.mean(rewards[-100:]) if rewards else 0.0
            avg_loss = np.mean(losses[-100:]) if losses else 0.0
            # 格式化输出，避免nan显示
            avg_loss = 0.0 if np.isnan(avg_loss) else avg_loss
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Avg Loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.2f}")
    # 保存模型（确保有loss才保存）
    if len(losses) > 0:
        agent.save("best_model.pth")
        print("训练完成，模型已保存为best_model.pth")
    else:
        print("训练完成，但未生成有效模型（经验池不足）")
    return agent

if __name__ == "__main__":
    # 初始化环境和智能体
    env = TicTacToe()
    agent = DQNAgent()

    # 训练智能体（可选，也可以直接加载预训练模型）
    print("开始训练...")
    agent = train_agent(env, agent, episodes=2000)
    agent.save("best_model.pth")
    print("训练完成，模型已保存。")

    # 开始人机对战
    play_human_vs_agent(env, agent)
