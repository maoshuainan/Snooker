import gymnasium as gym
import numpy as np
from snooker.oneredenv import OneRedEnv

env = OneRedEnv(render_mode="human")
obs, info = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print(obs)
env.close()