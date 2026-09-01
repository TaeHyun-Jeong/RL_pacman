from collections import deque
import numpy as np
import pygame

from Wall_Food_map import Wall, Food
from GhostPolicy import Ghost

class MyPacman:
    def __init__(self):
        self.width = 20     #가로 20칸
        self.height = 15    #세로 15칸
        self.wall = Wall()
        self.food = Food()
        self.ghost_policy = Ghost()
        self.max_ep = 500
        
        #reward 관련.
        self.food_reward = 1.0
        self.clear_reward = 10.0
        self.die_penalty = -10.0
        self.step_penalty = -0.005
        self.combo = 0
        self.combo_bonus = 0.1
        
        #action의 종류.
        self.num_actions = 4
        self.action = {
            0: (0, -1), #up
            1: (-1, 0), #left
            2: (0, 1),  #down
            3: (1, 0),  #right
        }
        
        #기타정보
        self.pacman_location = (1,1)
        self.ghost_location = (18,8)
        self.total_food = int(self.food.foodmap.sum())
        
        self.done = False
        self.steps = 0
        self.score = 0.0
        
        self.distance_bonus = 0.2
        self._phi_prev = None
        
        self.backtrack_penalty = -0.02
        self._prev_pos = None
        
        self.visit_penalty = -0.002
        self.visit_decay = 0.98
        self._visit = np.zeros((self.height, self.width), dtype=np.float32)
        
    def reset(self):
        self.pacman_location = (1,1)
        self.ghost_location = (18,8)
        if hasattr(self.ghost_policy, "reset"):
            self.ghost_policy.reset()
        self.done = False
        self.steps = 0
        self.score = 0.0
        self.combo = 0
        self.episode_food = self.food.foodmap.copy()    #에피소드에서 사용할 food map 복사.
        
        px, py = self.pacman_location#
        d0 = self._nearest_food_dist(px, py)
        self._phi_prev = 0.0 if d0 is None else -float(d0)
        
        self._prev_pos = self.pacman_location#
        self._visit.fill(0.0)
        
        return self._get_obs().astype(np.float32)
    
    def _get_obs(self):
        input_wall = self.wall.wallmap.flatten()  #wall을 그대로 입력으로
        input_food = self.episode_food.flatten()  #food를 그대로 입력으로
        H, W = self.height, self.width
        N = H * W

        pac   = np.zeros(N, dtype=np.float32)   #팩맨의 현재 위치를 나타내는 map
        px, py = self.pacman_location
        pac[py * W + px] = 1.0

        ghost = np.zeros(N, dtype=np.float32)   #Ghost의 현재 위치를 나타내는 map
        gx, gy = self.ghost_location
        ghost[gy * W + gx] = 1.0
        
        mode_now = self.ghost_policy._mode_at(self.ghost_policy.step_count)
        ghost_mode_flag = np.array([1.0 if mode_now == self.ghost_policy.CHASE else 0.0],dtype=np.float32)
        
        obs = np.concatenate([input_wall, input_food, pac, ghost, ghost_mode_flag])
        return obs.astype(np.float32)
    
    def step(self, p_action):
        self.steps += 1
        reward = 0.0
        px, py = self.pacman_location
        gx, gy = self.ghost_location
        
        def died(total_reward): #팩맨이 Ghost에게 잡혔을 때 끝내는 함수
            total_reward += self.die_penalty
            self.score += total_reward
            self.done = True
            return self._get_obs().astype(np.float32), float(total_reward), True, {}
    
        #팩맨을 action대로 이동.
        pdx, pdy = self.action.get(int(p_action), (0,0))
        tempx, tempy = px + pdx, py + pdy #일단 벽을 고려 안하고 이동한 위치를 계산.
        if (0 <= tempx < self.width and 0 <= tempy < self.height and self.wall.wallmap[tempy, tempx] == 0): #해당 위치가 벽인지 확인.
            mpx, mpy = tempx, tempy
        else:
            mpx, mpy = px, py
        self.pacman_location = (mpx, mpy)   #팩맨의 위치 업데이트
        if (mpx == gx and mpy == gy):   #이동한 위치가 Ghost에게 잡힌 위치인지 확인
            return died(reward)
            
        #food 먹기 처리
        if self.episode_food[mpy, mpx] == 1:
            self.episode_food[mpy, mpx] = 0
            reward += self.food_reward
            self.combo += 1
            reward += self.combo * self.combo_bonus
        else:
            self.combo = 0
        if (mpx, mpy) == self._prev_pos:
            reward += self.backtrack_penalty
        self._prev_pos = (px, py)  # '이전'은 이동 전 위치로 업데이트
        self._visit *= self.visit_decay
        vy, vx = mpy, mpx
        reward += self.visit_penalty * (1.0 + self._visit[vy, vx])
        self._visit[vy, vx] += 1.0
        #Ghost 이동
        g_action = self.ghost_policy.select_action(self)
        gdx, gdy = self.action[int(g_action)]
        mgx, mgy = gx + gdx, gy + gdy
        self.ghost_location  = (mgx, mgy)   #Ghost의 위치 업데이트
        if (mpx == mgx and mpy == mgy): #이동한 위치가 팩맨을 잡은 위치인지 확인
            return died(reward)
        px, py = self.pacman_location#
        d1 = self._nearest_food_dist(px, py)
        phi_now = 0.0 if d1 is None else -float(d1)
        reward += self.distance_bonus * (phi_now - self._phi_prev)
        self._phi_prev = phi_now
        #게임 종료인지 확인
        if self.episode_food.sum() == 0:
            reward += self.clear_reward
            self.done = True
            return self._get_obs().astype(np.float32), float(reward), bool(self.done), {}
        if self.steps >= self.max_ep:
            self.done = True
            return self._get_obs().astype(np.float32), float(reward), bool(self.done), {}
        
        reward += self.step_penalty
        self.score += reward
        
        obs = self._get_obs().astype(np.float32)
        return obs, float(reward), bool(self.done), {}
    
    def _nearest_food_dist(self, x, y):#
        if self.episode_food[y, x] == 1:
            return 0
        q = deque([(x, y, 0)])
        seen = set([(x, y)])
        while q:
            cx, cy, d = q.popleft()
            if self.episode_food[cy, cx] == 1:
                return d
            for dx, dy in self.action.values():
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height and self.wall.wallmap[ny, nx] == 0 and (nx, ny) not in seen:
                    seen.add((nx, ny)); q.append((nx, ny, d+1))
        return None  # 음식이 없다면 None
    
    def render(self, scale=28):
        W, H = self.width * scale, self.height * scale + 40
        if not hasattr(self, "_win"):
            self._win = pygame.display.set_mode((W, H))
            pygame.display.set_caption("MyPacman Environment")
            self._clock = pygame.time.Clock()
            self._font = pygame.font.SysFont("Arial", 22, bold=True)

        win = self._win
        win.fill((0, 0, 0))  # 배경: 검정

        # 색상 정의
        WALL  = (0, 0, 180)
        FOOD  = (255, 255, 255)
        PAC   = (255, 255, 0)
        GHOST = (255, 165, 0)
        EYEW  = (255, 255, 255)
        EYEB  = (0, 0, 255)

        def center(x, y):
            return (x * scale + scale // 2, y * scale + scale // 2)

        # 벽
        for y in range(self.height):
            for x in range(self.width):
                if self.wall.wallmap[y, x] == 1:
                    pygame.draw.rect(win, WALL, (x * scale, y * scale, scale, scale))

        # food
        pellet_r = max(3, scale // 10)
        for y in range(self.height):
            for x in range(self.width):
                if self.episode_food[y, x] == 1 and self.wall.wallmap[y, x] == 0:
                    cx, cy = center(x, y)
                    pygame.draw.circle(win, FOOD, (cx, cy), pellet_r)

        # Pac-Man (노란 원)
        px, py = self.pacman_location
        pcx, pcy = center(px, py)
        pygame.draw.circle(win, PAC, (pcx, pcy), int(scale * 0.45))

        # Ghost (오리지널 모양)
        gx, gy = self.ghost_location
        gcx, gcy = center(gx, gy)
        gw = int(scale * 0.8)
        gh = int(scale * 0.9)
        top = gcy - gh // 2
        left = gcx - gw // 2

        # 몸통 (상단 반원 + 하단 물결)
        body_points = []
        num_waves = 3
        for i in range(num_waves + 1):
            wx = left + i * (gw / num_waves)
            wy = top + gh * 0.9 + (scale * 0.05 * ((i % 2) * 2 - 1))
            body_points.append((wx, wy))
        body_points = [(left, top + gh * 0.5)] + body_points + [(left + gw, top + gh * 0.5)]

        pygame.draw.circle(win, GHOST, (gcx, top + gh // 2), gw // 2)  # 반원
        pygame.draw.polygon(win, GHOST, body_points)  # 하단 물결

        # 눈 (정면 고정)
        eye_offset_x = int(gw * 0.22)
        eye_offset_y = int(gh * 0.2)
        eye_r = max(4, scale // 10)
        pupil_r = max(2, scale // 20)

        # 왼쪽 눈
        lx, ly = gcx - eye_offset_x, top + gh * 0.4
        pygame.draw.circle(win, EYEW, (int(lx), int(ly)), eye_r)
        pygame.draw.circle(win, EYEB, (int(lx), int(ly)), pupil_r)

        # 오른쪽 눈
        rx, ry = gcx + eye_offset_x, top + gh * 0.4
        pygame.draw.circle(win, EYEW, (int(rx), int(ry)), eye_r)
        pygame.draw.circle(win, EYEB, (int(rx), int(ry)), pupil_r)

        # 점수 표시
        eaten_food = int(self.total_food - int(self.episode_food.sum()))
        score_text = self._font.render(f"SCORE: {eaten_food}", True, (255, 255, 0))
        win.blit(score_text, (10, self.height * scale + 5))

        pygame.display.flip()
        self._clock.tick(30)