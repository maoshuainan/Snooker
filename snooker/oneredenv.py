import numpy as np

import gymnasium as gym
from gymnasium import spaces

from .settings import *
from .game import *
from snooker import balls
import math
from .vec2D import Vec2d as Vec2D
from collections import deque
from .cue import Cue
from .score import Score
from .draw import Draw

class OneRedEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(self, render_mode=None, time_limit = 100):
        self.window_size = SCREEN_SIZE  # The size of the PyGame window
        self.time_limit = time_limit
        
        _low_boundeds = np.array([100,50])
        _high_boundeds = np.array([1000,523])
        self.observation_space = spaces.Tuple((spaces.Box(_low_boundeds, _high_boundeds, dtype=np.float32),
                                              spaces.Box(_low_boundeds, _high_boundeds, dtype=np.float32),))

        # We have 2 actions, corresponding to "angle", "force"
        self.action_space = spaces.Dict(
            {
                "angle": spaces.Discrete(360, start = 0),
                "force": spaces.Discrete(91, start = 20)
            }
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.screen` will be a reference
        to the screen that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.screen = None
        self.clock = None

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.game = OneRedGame()
        self.time = 0
        observation = self.game.get_balls_coords()
        info = {}
        
        if self.render_mode == "human":
            self._render_frame()
            
        return observation, info
    
    def step(self, action):
        self.time += 1
        self.start_pos, self.end_pos = self.game.cue_handler(action)
        
        while True:
            self.game.game_handler()
            if self.render_mode == "human":
                self._render_frame()
                
            if self.game.board_status == STATICK:
                break
        
        terminated = self.game.state_evaluate()
        truncated = self.time > self.time_limit
        observation = self.game.get_balls_coords()
        reward = self.game.turn.points
        info = {}
        
        return observation, reward, terminated, truncated, info
        
    def render(self):
        if self.render_mode == "rgb_array":
            self._render_frame()
    
    def _render_background(self):
        self.game.painter.game_surface.fill(BACKGROUND_COLOR)
        self.game.painter.game_surface.blit(self.game.table, TABLE_POS)
        self.game.painter.game_surface.blit(self.game.score.score_board, SCORE)
    
    def _render_balls(self):
        for ball in self.game.all_balls:
            if ball.vizibility:
                self.game.painter.draw_balls(ball)

    def _render_cue(self):
        self.game.painter.cue_draw(self.game.cue, self.start_pos, self.end_pos)

    def _render_frame(self):
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode(SCREEN_SIZE)
            pygame.display.set_caption('Cool Snooker')
        
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         return False
        
        self._render_background()
        if self.game.hit:
            self._render_cue()
        self._render_balls()    
        self.screen.blit(self.game.painter.game_surface, (0, 0))
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
        
    def close(self):
        pygame.quit()
    
class OneRedGame():
    """
    The class is constructed for simple one red game.
    """
    pockets = {'ur_pocket': [Vec2D(UR_POCKET), Vec2D(125, 94), Vec2D(113, 78),
                             Vec2D(143, 75), Vec2D(128, 63)],
               'ul_pocket': [Vec2D(UL_POCKET), Vec2D(125, 480),
                             Vec2D(113, 495), Vec2D(143, 498),
                             Vec2D(128, 510)],
               'dl_pocket': [Vec2D(DL_POCKET), Vec2D(974, 480),
                             Vec2D(986, 495), Vec2D(956, 498),
                             Vec2D(971, 510)],
               'dr_pocket': [Vec2D(DR_POCKET), Vec2D(956, 75), Vec2D(971, 63),
                             Vec2D(974, 94), Vec2D(986, 79)],
               'ml_pocket': [Vec2D(ML_POCKET), Vec2D(530, 498),
                             Vec2D(539, 510), Vec2D(568, 498),
                             Vec2D(560, 510)],
               'mr_pocket': [Vec2D(MR_POCKET), Vec2D(530, 75), Vec2D(539, 63),
                             Vec2D(568, 75), Vec2D(560, 63)]}
    
    def __init__(self):
        self.moving_balls = deque([])
        self.hitted_balls = deque([])
        self.potted = []
        self.table = pygame.image.load('./snooker/Snooker_table3.png')
        self.white_ball = balls.WhiteBall(coords=POS_WHITE)
        self.redball1 = balls.RedBall(coords=POS_RED1)
        self.first_player = Player("O'Sullivan")
        self.second_player = Player("Selby")
        self.all_balls = deque([
                               self.redball1,
                               self.white_ball
                               ])
        self.cue = Cue()
        self.turn = self.first_player
        self.board_status = STATICK
        self.score = Score()
        self.foul = False
        self.hit = False
        self.painter = Draw()
    
    def game_handler(self):
        # 展示分数
        self.score.show_score(self.first_player, self.second_player, self.turn)
        # 对所有球进行碰撞判定
        self.ball_update()
        # 对所有球进行位置更新
        self.balls_handler()
        # 判断桌面是否是静止（STATICK）状态
        self.if_statick_board()
    
    def state_evaluate(self):
        # 初始化犯规情况
        self.foul = False
        # 没打中球的情况
        if not self.hitted_balls and self.hit is True and not self.potted:
            self.turn.points -= FOUL_POINTS
            # print("Foul no ball hit")
        # 等待击球
        self.hit = False
        # 如果落袋
        if self.potted:
            for ball in self.potted:
                if isinstance(ball, balls.WhiteBall):
                    self.turn.points -= FOUL_POINTS
                if isinstance(ball, balls.RedBall):
                    self.turn.points += ball.points
            return True
        return False
        
    def if_statick_board(self):
        for ball in self.all_balls:
            if ball.velocity.length > 0 and ball not in self.moving_balls:
                self.moving_balls.append(ball)
            elif ball in self.moving_balls and ball.velocity.length == 0:
                self.moving_balls.remove(ball)
        if not self.moving_balls:
            self.board_status = STATICK
        else:
            self.board_status = NON_STATICK
            
    def cue_handler(self, action):
        angle = action["angle"]
        r2 = action["force"]
        start_pos = Vec2D(0, 0)
        end_pos = Vec2D(0, 0)
        r = r2 + 160
        if r < CUE_DEFAULT_R:
            r = CUE_DEFAULT_R
        if r > CUE_DEFAULT_R + CUE_DEFAULT_R / 2:
            r = CUE_DEFAULT_R + CUE_DEFAULT_R / 2
        end_pos.x = self.white_ball.coords.x + math.cos(math.radians(angle)) * r
        end_pos.y = self.white_ball.coords.y + math.sin(math.radians(angle)) * r
        start_pos.x = self.white_ball.coords.x + math.cos(math.radians(angle)) * r2
        start_pos.y = self.white_ball.coords.y + math.sin(math.radians(angle)) * r2
        new_velocity = Vec2D.normalized(start_pos - end_pos)
        force = Vec2D(self.white_ball.coords - start_pos).length
        self.white_ball.velocity = new_velocity *\
            force ** 2 / MIN_HITTING_FORCE
        self.hit = True
        return start_pos, end_pos
        
        
    def ball_update(self):
        for a in range(0, len(self.all_balls)-1):
            for b in range(a+1, len(self.all_balls)):
                ball, next_ball = self.all_balls[a], self.all_balls[b]
                delta = next_ball.coords - ball.coords
                if (next_ball.coords - ball.coords).length <= ball.RADIUS * 2:
                    if ball.velocity.length > 0 and \
                            next_ball.velocity.length > 0:
                        ball.coords += Vec2D.normalized(delta) *\
                            (delta.length - ball.RADIUS * 2)
                        next_ball.coords += Vec2D.normalized(-delta) *\
                            (delta.length - ball.RADIUS * 2)
                        self.ball_collision(ball, next_ball)
                    elif ball.velocity.length > 0:
                        if isinstance(ball, balls.WhiteBall):
                            self.hitted_balls.append(next_ball)
                        ball.coords += Vec2D.normalized(delta) *\
                            (delta.length - ball.RADIUS * 2)
                        self.ball_collision(ball, next_ball)
                    elif next_ball.velocity.length > 0:
                        if isinstance(next_ball, balls.WhiteBall):
                            self.hitted_balls.append(ball)
                        next_ball.coords += Vec2D.normalized(-delta) *\
                            (delta.length - ball.RADIUS * 2)
                        self.ball_collision(ball, next_ball)
    
    def ball_collision(self, ball, next_ball):
        delta = next_ball.coords - ball.coords
        unit_delta = Vec2D(delta/delta.get_length())
        unit_tangent = Vec2D(unit_delta.perpendicular())
        velocity_1_n = unit_delta.dot(ball.velocity)
        velocity_1_t = unit_tangent.dot(ball.velocity)
        velocity_2_n = unit_delta.dot(next_ball.velocity)
        velocity_2_t = unit_tangent.dot(next_ball.velocity)
        new_velocity_1_t = velocity_1_t
        new_velocity_2_t = velocity_2_t
        new_velocity_1_n = velocity_2_n
        new_velocity_2_n = velocity_1_n
        new_velocity_1_n = Vec2D(new_velocity_1_n * unit_delta)
        new_velocity_2_n = Vec2D(new_velocity_2_n * unit_delta)
        new_velocity_1_t = Vec2D(new_velocity_1_t * unit_tangent)
        new_velocity_2_t = Vec2D(new_velocity_2_t * unit_tangent)
        new_velocity_1 = Vec2D(new_velocity_1_n + new_velocity_1_t)
        new_velocity_2 = Vec2D(new_velocity_2_n + new_velocity_2_t)
        ball.velocity = new_velocity_1
        next_ball.velocity = new_velocity_2
    
    def balls_handler(self):
        for ball in self.all_balls:
            if ball.velocity.length > 0:
                ball.move(self.pockets)
            if ball.is_potted and ball not in self.potted:
                self.potted.append(ball)
    
    def get_balls_coords(self):
        return tuple([x.coords for x in self.all_balls])
        