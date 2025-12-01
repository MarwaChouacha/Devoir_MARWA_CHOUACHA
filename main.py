from gridworld_env import GridWorldEnv
from agent import RandomAgent
import time

env = GridWorldEnv(render_mode="human")
agent = RandomAgent(env.action_space)

state, info = env.reset()

for _ in range(20):
    action = agent.act(state)
    state, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(0.2)

    if terminated:
        print("Goal reached!")
        break

env.render()
