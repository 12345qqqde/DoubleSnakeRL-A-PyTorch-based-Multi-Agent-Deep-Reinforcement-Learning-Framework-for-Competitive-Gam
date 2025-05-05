# 导入必要的库
import numpy as np  # 数值计算库
import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
import torch.optim as optim  # 优化器模块
import torch.nn.functional as F  # 神经网络函数模块
from collections import deque  # 双端队列（用于经验回放缓冲区）


class DQN(nn.Module):
    """深度Q网络（Deep Q-Network）模型"""

    def __init__(self, input_size, hidden_size, output_size):
        """
        初始化DQN模型
        :param input_size: 输入层维度（状态空间大小）
        :param hidden_size: 隐藏层维度
        :param output_size: 输出层维度（动作空间大小）
        """
        super(DQN, self).__init__()  # 调用父类初始化
        # 网络层定义
        self.fc1 = nn.Linear(input_size, hidden_size)  # 输入层到隐藏层1
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # 隐藏层1到隐藏层2
        self.fc3 = nn.Linear(hidden_size, hidden_size)  # 隐藏层2到隐藏层3
        self.fc4 = nn.Linear(hidden_size, output_size)  # 隐藏层3到输出层

    def forward(self, x):
        """前向传播过程"""
        x = F.relu(self.fc1(x))  # 第一层+ReLU激活
        x = F.relu(self.fc2(x))  # 第二层+ReLU激活
        x = F.relu(self.fc3(x))  # 第三层+ReLU激活
        x = self.fc4(x)  # 输出层（无激活函数）
        return x


class SnakeAI:
    """双蛇AI控制类，使用深度Q学习算法"""

    def __init__(self, buffer_size=1000, batch_size=32):
        """
        初始化AI参数
        :param buffer_size: 经验回放缓冲区大小
        :param batch_size: 训练批大小
        """
        # 超参数设置
        self.gamma = 0.90  # 折扣因子（未来奖励衰减系数）
        self.input_size = 24  # 输入特征维度（24维状态向量）
        self.output_size = 8  # 输出动作维度（前4个是蛇1动作，后4个是蛇2动作）
        self.hidden_size = 120  # 隐藏层神经元数量
        self.discount_factor = 0.90  # 目标网络更新折扣因子

        # 神经网络初始化
        self.model = DQN(self.input_size, self.hidden_size, self.output_size)  # 主网络
        self.target_model = DQN(self.input_size, self.hidden_size, self.output_size)  # 目标网络
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)  # Adam优化器
        self.loss_fn = nn.MSELoss()  # 均方误差损失函数

        # 经验回放缓冲区
        self.buffer = deque(maxlen=buffer_size)  # 使用双端队列实现循环缓冲区
        self.batch_size = batch_size

        # 同步目标网络权重
        self.target_model.load_state_dict(self.model.state_dict())

    def train_model(self):
        """训练神经网络模型"""
        if len(self.buffer) < self.batch_size:
            return

        batch_indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[idx] for idx in batch_indices]

        states = np.array([sample[0] for sample in batch], dtype=np.float32)
        actions = np.array([sample[1] for sample in batch])
        rewards = np.array([sample[2] for sample in batch], dtype=np.float32)
        next_states = np.array([sample[3] for sample in batch], dtype=np.float32)
        dones = np.array([sample[4] for sample in batch], dtype=np.float32)

        states_tensor = torch.FloatTensor(states)
        next_states_tensor = torch.FloatTensor(next_states)
        rewards_tensor = torch.FloatTensor(rewards)
        dones_tensor = torch.FloatTensor(dones)

        current_q = self.model(states_tensor)

        with torch.no_grad():
            next_q = self.model(next_states_tensor)

        snake1_next_q = next_q[:, :4]
        snake2_next_q = next_q[:, 4:]

        snake1_max_q = torch.max(snake1_next_q, dim=1)[0]
        snake2_max_q = torch.max(snake2_next_q, dim=1)[0]

        targets_snake1 = rewards_tensor + self.gamma * snake1_max_q * (1 - dones_tensor)
        targets_snake2 = rewards_tensor + self.gamma * snake2_max_q * (1 - dones_tensor)
        targets = torch.stack([targets_snake1, targets_snake2], dim=1)

        target_q = current_q.clone().detach()

        actions_snake1 = torch.LongTensor(actions[:, 0])
        actions_snake2 = torch.LongTensor(actions[:, 1])
        batch_indices = torch.arange(self.batch_size)

        target_q[batch_indices, actions_snake1] = targets[:, 0]
        target_q[batch_indices, actions_snake2 + 4] = targets[:, 1]

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        """更新目标网络权重（硬更新）"""
        self.target_model.load_state_dict(self.model.state_dict())

    def add_experience(self, state, action, reward, next_state, done):
        """添加经验到回放缓冲区"""
        self.buffer.append((state, action, reward, next_state, done))

    def get_action(self, state):
        """根据当前状态选择动作（ε-贪婪策略）"""
        state = np.array(state, dtype=np.float32)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze().numpy()

        snake1_action = np.argmax(q_values[:4])
        snake2_action = np.argmax(q_values[4:])

        return [snake1_action, snake2_action]
