import pygame
import torch
import numpy as np
import random
from Game import Snake1, Snake2, Food, SCREEN_WIDTH, SCREEN_HEIGHT, BLOCK_SIZE, POP_SIZE, WHITE
from network import SnakeAI

# 颜色定义
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
PURPLE = (128, 0, 128)

# 初始化pygame
pygame.init()
font = pygame.font.SysFont('comics', 30)

class Game:
    def __init__(self, buffer_size=10000, batch_size=64):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.snake1 = Snake1()
        self.snake2 = Snake2()
        self.food_count = random.randint(1, 5)  # 随机生成1到5个食物
        self.food = Food(self.food_count)
        self.ai_player = SnakeAI(buffer_size, batch_size)
        
        # 加载预训练模型
        try:
            state_dict = torch.load('best_weights.pth', weights_only=True)
            self.ai_player.model.load_state_dict(state_dict)
        except Exception as e:
            print(f"加载权重时出现错误: {e}")

        # 初始化统计数据
        self.scores1 = []
        self.scores2 = []
        self.best_score = 0
        self.total_food_eaten = 0
        self.current_episode = 0
        self.i1 = 0  # 蛇1步数计数器
        self.i2 = 0  # 蛇2步数计数器

    def update(self, ai_player, tran_i1, tran_i2):
        state = self.get_state()
        action = ai_player.get_action(state)

        # 处理双蛇移动方向
        v_a1 = self.snake1.direction
        v_a2 = self.snake2.direction
        v_b1, v_b2 = self.get_direction(action)
        
        self.snake1.turn(v_b1)
        self.snake2.turn(v_b2)

        # 重置步数计数器
        if v_a1 != v_b1 and (v_a1[0] * -1, v_a1[1] * -1) != v_b1:
            self.i1 = 0
        if v_a2 != v_b2 and (v_a2[0] * -1, v_a2[1] * -1) != v_b2:
            self.i2 = 0

        # 计算距离
        head1 = np.array(self.snake1.get_head_position())
        head2 = np.array(self.snake2.get_head_position())
        food_pos = np.array(self.food.positions[0])  # 使用第一个食物的位置进行距离计算
        distances1 = np.linalg.norm(head1 - food_pos)
        distances2 = np.linalg.norm(head2 - food_pos)

        # 移动双蛇
        self.snake1.move()
        self.snake2.move()

        done = False
        # 吃食物检测
        if self.food.check_eaten_by_snake1(self.snake1):
            self.snake1.length += 1
            if not self.food.positions:
                self.food_count = random.randint(1, 5)
                self.food = Food(self.food_count)
        if self.food.check_eaten_by_snake2(self.snake2):
            self.snake2.length += 1
            if not self.food.positions:
                self.food_count = random.randint(1, 5)
                self.food = Food(self.food_count)

        # 碰撞检测
        if self.is_collision():
            self.scores1.append(self.snake1.length)
            self.scores2.append(self.snake2.length)
            done = True

        # 计算奖励
        next_state = self.get_state()
        reward1 = self.get_reward_snake1(done, distances1, tran_i1)
        reward2 = self.get_reward_snake2(done, distances2, tran_i2)
        reward = reward1 + reward2

        # 经验回放
        ai_player.add_experience(state, action, reward, next_state, done)
        ai_player.train_model()
        return done

    def get_reward_snake1(self, done, distances, tran_i1):
        head = np.array(self.snake1.get_head_position())
        new_dist = np.linalg.norm(head - np.array(self.food.positions[0])) if self.food.positions else distances
        reward = 0
        if tran_i1 > 50:
            reward -= 10
        if done:
            reward -= 200
        elif any(f == self.snake1.get_head_position() for f in self.food.positions):
            reward += 100
            self.total_food_eaten += 1
        elif new_dist < distances:
            reward += 10
        else:
            reward -= 1
        return reward

    def get_reward_snake2(self, done, distances, tran_i2):
        head = np.array(self.snake2.get_head_position())
        new_dist = np.linalg.norm(head - np.array(self.food.positions[0])) if self.food.positions else distances
        reward = 0
        if tran_i2 > 50:
            reward -= 10
        if done:
            reward -= 200
        elif any(f == self.snake2.get_head_position() for f in self.food.positions):
            reward += 100
            self.total_food_eaten += 1
        elif new_dist < distances:
            reward += 10
        else:
            reward -= 1
        return reward

    def is_collision(self):
        head1 = self.snake1.get_head_position()
        head2 = self.snake2.get_head_position()
        collision1 = head1 in self.snake1.positions[1:] or head1 in self.snake2.positions
        collision2 = head2 in self.snake2.positions[1:] or head2 in self.snake1.positions
        return collision1 or collision2

    def get_direction(self, actions):
        snake1_dir = [(0, -1), (0, 1), (-1, 0), (1, 0)][actions[0]]
        snake2_dir = [(0, -1), (0, 1), (-1, 0), (1, 0)][actions[1]]
        return snake1_dir, snake2_dir

    def get_state(self):
        head1 = self.snake1.get_head_position()
        head2 = self.snake2.get_head_position()
        food_pos = self.food.positions[0] if self.food.positions else (0, 0)
        
        state = [
            # 蛇1的障碍检测
            (head1[0] - BLOCK_SIZE, head1[1]) in self.snake1.positions[1:] + self.snake2.positions,
            (head1[0] + BLOCK_SIZE, head1[1]) in self.snake1.positions[1:] + self.snake2.positions,
            (head1[0], head1[1] - BLOCK_SIZE) in self.snake1.positions[1:] + self.snake2.positions,
            (head1[0], head1[1] + BLOCK_SIZE) in self.snake1.positions[1:] + self.snake2.positions,
            # 蛇2的障碍检测
            (head2[0] - BLOCK_SIZE, head2[1]) in self.snake2.positions[1:] + self.snake1.positions,
            (head2[0] + BLOCK_SIZE, head2[1]) in self.snake2.positions[1:] + self.snake1.positions,
            (head2[0], head2[1] - BLOCK_SIZE) in self.snake2.positions[1:] + self.snake1.positions,
            (head2[0], head2[1] + BLOCK_SIZE) in self.snake2.positions[1:] + self.snake1.positions,
            # 食物位置信息
            food_pos[0] < head1[0], food_pos[0] > head1[0], food_pos[1] < head1[1], food_pos[1] > head1[1],
            food_pos[0] < head2[0], food_pos[0] > head2[0], food_pos[1] < head2[1], food_pos[1] > head2[1],
            # 移动方向
            *[d == self.snake1.direction for d in [(0, -1), (0, 1), (-1, 0), (1, 0)]],
            *[d == self.snake2.direction for d in [(0, -1), (0, 1), (-1, 0), (1, 0)]]]
        return np.array(state, dtype=np.float32)

    def run(self):
        pygame.display.set_caption("Double Snake AI Trainer")
        for episode in range(POP_SIZE):
            self.current_episode = episode + 1
            self.snake1.reset()
            self.snake2.reset()
            self.food_count = random.randint(1, 5)
            self.food = Food(self.food_count)
            done = False
            
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()
                
                self.i1 += 1
                self.i2 += 1
                done = self.update(self.ai_player, self.i1, self.i2)
                
                self.screen.fill(WHITE)
                self.snake1.draw(self.screen)
                self.snake2.draw(self.screen)
                self.food.draw(self.screen)

                info_texts = [
                    f"Episode: {self.current_episode}/{POP_SIZE}",
                    f"Snake1: {self.snake1.length} (Best: {max(self.scores1 or [0])})",
                    f"Snake2: {self.snake2.length} (Best: {max(self.scores2 or [0])})",
                    f"Total Food: {self.total_food_eaten}",
                    f"Global Best: {self.best_score}"
                ]
                for i, text in enumerate(info_texts):
                    text_surf = font.render(text, True, [BLACK, GREEN, BLUE, PURPLE, RED][i])
                    self.screen.blit(text_surf, (10, 10 + i * 30))

                pygame.display.flip()
                self.clock.tick(20)

            current_max = max(self.snake1.length, self.snake2.length)
            if current_max > self.best_score:
                self.best_score = current_max
                torch.save(self.ai_player.model.state_dict(), 'best_weights.pth')

            self.total_food_eaten += (self.snake1.length - 3) + (self.snake2.length - 3)
            print(f"Episode {self.current_episode} Completed | Snake1: {self.snake1.length} | Snake2: {self.snake2.length}")

if __name__ == "__main__":
    game = Game()
    game.run()
