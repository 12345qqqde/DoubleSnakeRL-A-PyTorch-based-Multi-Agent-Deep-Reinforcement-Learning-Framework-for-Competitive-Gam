import pygame
import sys
import random
# 定义常量：窗口尺寸，像素单位EZ
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BLOCK_SIZE = 20
POP_SIZE = 10000  # 运行总次数
# 定义颜色
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
PURPLE = (128, 0, 128)  # 紫色
# 全局变量
score_snake1 = 0
score_snake2 = 0

class Snake:
    # 初始化（）
    # 初始化蛇的长度（length），蛇的位置（positions），蛇的移动方向（direction），蛇的颜色（color = Green）
    def __init__(self):
        self.length = 3
        self.positions = [(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)]  # 初始化在正中间
        self.direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])  # 四种方向
        self.color = GREEN  # (默认颜色为绿色)

    # 获得蛇头的坐标（）
    def get_head_position(self):
        return self.positions[0]

    # 改变蛇移动方向（point：改变方向）
    #   如果改变的方向和蛇的原方向相反，则蛇的方向不改变
    #   否则，改变蛇的移动方向
    def turn(self, point):
        if (point[0] * -1, point[1] * -1) == self.direction:
            return
        else:
            self.direction = point

    # 移动（）
    #   根据当前方向计算下一个位置
    def move(self):
        cur = self.get_head_position()
        x, y = self.direction  # 解包赋值，获得偏移量
        new = ((cur[0] + (x * BLOCK_SIZE)) % SCREEN_WIDTH, (cur[1] + (y * BLOCK_SIZE)) % SCREEN_HEIGHT)  # BLOCK_SIZE 是每一格的尺寸大小，计算蛇头的新的坐标
        self.positions.insert(0, new)  # 将新的蛇头插入到原来位置的0索引处。
        if len(self.positions) > self.length:
            self.positions.pop()  # 移除蛇尾位置信息

    # 重新开始（）
    def reset(self):
        self.length = 3  # 初始长度为3
        self.positions = [(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)]
        self.direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])

    # 画蛇（surface：窗口对象）
    #   遍历蛇身的位置，将其画在画布上
    def draw(self, surface):
        for p in self.positions:
            r = pygame.Rect((p[0], p[1]), (BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(surface, self.color, r)


class Snake1(Snake):
    def __init__(self):
        super().__init__()
        # 颜色定义为红色
        self.color = RED

    def reset(self):
        self.length = 3  # 初始长度为3
        self.positions = [(BLOCK_SIZE, BLOCK_SIZE)]
        self.direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])

    def update_score(self):
        global score_snake1  # 引用全局变量
        score_snake1 += 1
        self.length += 1
        return score_snake1

    def get_distance_to_snake2(self, snake2):
        # 检测蛇头和蛇身之间的的欧几里得距离。
        head1 = self.get_head_position()
        min_distance = float('inf')  # 初始化舍身最小值变量为无穷大
        for pos in snake2.positions:
            distance = ((head1[0] - pos[0]) ** 2 + (head1[1] - pos[1]) ** 2) ** 0.5
            min_distance = min(min_distance, distance)
        return min_distance

    def check_collision_with_snake2(self, snake2):
        head1 = self.get_head_position()
        for pos in snake2.positions:
            if head1 == pos:  # 遍历蛇身位置，如果有蛇头和蛇2身重合则返回TURE
                return True
        return False


class Snake2(Snake):
    def __init__(self):
        super().__init__()
        self.color = BLUE  # 初始化为蓝色

    def reset(self):
        self.length = 3  # 初始长度为3
        self.positions = [(SCREEN_HEIGHT - 3 * BLOCK_SIZE, SCREEN_HEIGHT - 3 * BLOCK_SIZE)]
        self.direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])

    def update_score(self):
        global score_snake2
        score_snake2 += 1
        self.length += 1
        return score_snake2

    def get_distance_to_snake1(self, snake1):
        head2 = self.get_head_position()
        min_distance = float('inf')  # 初始化舍身最小值变量为无穷大
        for pos in snake1.positions:
            distance = ((head2[0] - pos[0]) ** 2 + (head2[1] - pos[1]) ** 2) ** 0.5
            min_distance = min(min_distance, distance)
        return min_distance

    def check_collision_with_snake1(self, snake1):
        head2 = self.get_head_position()
        for pos in snake1.positions:
            if head2 == pos:  # 遍历蛇身位置，如果有蛇头和蛇1身重合则返回TURE
                return True
        return False


class Food:
    def __init__(self, count=1):
        self.count = count  # 食物数量
        self.positions = []  # 食物位置列表
        self.color = GREEN
        self.reset()

    def reset(self):
        self.positions = []
        for _ in range(self.count):
            x = random.randrange(0, SCREEN_WIDTH, BLOCK_SIZE)
            y = random.randrange(0, SCREEN_HEIGHT, BLOCK_SIZE)
            self.positions.append((x, y))

    def draw(self, surface):
        for pos in self.positions:
            r = pygame.Rect((pos[0], pos[1]), (BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(surface, self.color, r)
            pygame.draw.rect(surface, BLACK, r, 1)

    def check_eaten_by_snake1(self, snake1):
        head = snake1.get_head_position()
        if head in self.positions:
            self.positions.remove(head)
            return True
        return False

    def check_eaten_by_snake2(self, snake2):
        head = snake2.get_head_position()
        if head in self.positions:
            self.positions.remove(head)
            return True
        return False
