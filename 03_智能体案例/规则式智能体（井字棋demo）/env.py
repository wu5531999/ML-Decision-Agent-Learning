import numpy as np

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)  # 0: empty, 1: player1, -1: player2
        self.current_player = 1  # 1 或 -1

    def reset(self):
        """重置棋盘到初始状态"""
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        return self.get_state()

    def get_state(self):
        """返回9维向量表示棋盘状态：己方=1, 对方=-1, 空=0"""
        return self.board.flatten()

    def get_legal_actions(self):
        """返回所有合法动作索引（0-8）"""
        return np.where(self.board.flatten() == 0)[0].tolist()

    def check_win(self):
        """检查当前玩家是否获胜，返回：1（胜）、-1（负）、0（平局）、None（未结束）"""
        # 行、列、对角线
        for i in range(3):
            if abs(np.sum(self.board[i, :])) == 3:
                return 1 if self.board[i, 0] == 1 else -1
            if abs(np.sum(self.board[:, i])) == 3:
                return 1 if self.board[0, i] == 1 else -1
        if abs(np.sum(self.board.diagonal())) == 3:
            return 1 if self.board[0, 0] == 1 else -1
        if abs(np.sum(np.fliplr(self.board).diagonal())) == 3:
            return 1 if self.board[0, 2] == 1 else -1
        # 平局
        if len(self.get_legal_actions()) == 0:
            return 0
        # 游戏未结束
        return None

    def step(self, action):
        """
        执行一步动作
        :param action: 0-8 的整数，对应棋盘位置
        :return: next_state, reward, done, info
        """
        if action not in self.get_legal_actions():
            raise ValueError(f"非法动作 {action}，当前棋盘：\n{self.board}")

        row, col = action // 3, action % 3
        self.board[row, col] = self.current_player

        win = self.check_win()
        if win is not None:
            done = True
            if win == 1:
                reward = 1
            elif win == -1:
                reward = -1
            else:
                reward = 0
        else:
            self.current_player *= -1
            done = False
            reward = 0

        return self.get_state(), reward, done, {"win": win}

def print_board_with_1_3_labels(board):
    """打印带1-3行列标签的棋盘"""
    print("   1   2   3")  # 列标签
    print("  ──────────")
    for i in range(3):
        row_str = f"{i+1} "  # 行标签（1-3）
        for j in range(3):
            val = board[i][j]
            if val == 1:
                row_str += "| X "
            elif val == -1:
                row_str += "| O "
            else:
                row_str += "|   "
        row_str += "|"
        print(row_str)
        print("  ──────────")

def test_random_games(env, n=100):
    """随机对局n次，统计胜率和平局率"""
    results = {"win1": 0, "win2": 0, "draw": 0}
    for _ in range(n):
        state = env.reset()
        done = False
        while not done:
            legal_actions = env.get_legal_actions()
            action = np.random.choice(legal_actions)
            state, reward, done, info = env.step(action)
        if info["win"] == 1:
            results["win1"] += 1
        elif info["win"] == -1:
            results["win2"] += 1
        else:
            results["draw"] += 1
    print(f"随机对局 {n} 次结果：")
    print(f"玩家1胜：{results['win1']}, 玩家2胜：{results['win2']}, 平局：{results['draw']}")

def play_episode(env, agent, replay_buffer=None, epsilon=0.1):
    """
    玩一局游戏，收集经验存入replay_buffer
    :param env: TicTacToe环境
    :param agent: DQN智能体
    :param replay_buffer: 经验回放缓冲区
    :param epsilon: e-greedy探索率
    :return: total_reward
    """
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        # e-greedy选择动作
        if np.random.rand() < epsilon:
            action = np.random.choice(env.get_legal_actions())
        else:
            action = agent.select_action(state, env.get_legal_actions())

        next_state, reward, done, info = env.step(action)
        if replay_buffer is not None:
            replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    return total_reward

def play_human_vs_agent(env, agent):
    """
    人机对战命令行脚本
    玩家输入坐标（如 1 1 表示第1行第1列，从1开始计数）
    """
    print("欢迎来到井字棋人机对战！")
    print("棋盘坐标：行 列，例如 1 1 表示左上角，3 3 表示右下角")
    print("玩家执X（1），智能体执O（-1）\n")

    while True:
        state = env.reset()
        done = False
        while not done:
            if env.current_player == 1:
                # 玩家回合
                print("当前棋盘：")
                print_board_with_1_3_labels(env.board)
                while True:
                    try:
                        # 接收用户输入的 1-3 范围坐标
                        row_input, col_input = map(int, input("请输入你的落子位置（行 列，范围1-3）：").split())
                        # 转换为 0-2 的索引
                        row = row_input - 1
                        col = col_input - 1
                        # 校验输入范围
                        if row not in [0,1,2] or col not in [0,1,2]:
                            print("坐标超出范围！请输入 1-3 之间的数字")
                            continue
                        action = row * 3 + col
                        if action in env.get_legal_actions():
                            break
                        else:
                            print("该位置已有棋子，请重新输入！")
                    except:
                        print("输入格式错误，请输入两个数字，如 1 1")
            else:
                # 智能体回合
                legal_actions = env.get_legal_actions()
                action = agent.select_action(state, legal_actions)
                row, col = action // 3, action % 3
                # 智能体落子位置也转换为 1-3 显示
                print(f"智能体选择：行 {row+1} 列 {col+1}")

            state, reward, done, info = env.step(action)

            if done:
                print("最终棋盘：")
                print_board_with_1_3_labels(env.board)
                if info["win"] == 1:
                    print("🎉 恭喜你赢了！")
                elif info["win"] == -1:
                    print("💻 智能体赢了！")
                else:
                    print("🤝 平局！")
                break

        again = input("再来一局？(y/n)：")
        if again.lower() != "y":
            break

if __name__ == "__main__":
    # 测试游戏环境
    env = TicTacToe()
    test_random_games(env, 100)