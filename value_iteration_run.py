# value_iteration_run.py
import numpy as np
import matplotlib.pyplot as plt
from gridworld_env import GridWorldEnv
import time

# ----------------------------
# Helpers: mapping state <-> index
# ----------------------------
def pos_to_index(pos, grid_size):
    """pos: array-like [r,c] or tuple"""
    r, c = int(pos[0]), int(pos[1])
    return r * grid_size + c

def index_to_pos(idx, grid_size):
    r = idx // grid_size
    c = idx % grid_size
    return np.array([r, c], dtype=np.int32)

# ----------------------------
# Transition simulator (no side-effects)
# Must replicate GridWorldEnv.step logic
# ----------------------------
def simulate_step(env, state_pos, action):
    """
    Simule la transition à partir d'une position sans modifier env.
    Retourne: next_pos (np.array), reward (float), terminated (bool)
    """
    grid_size = env.grid_size
    moves = {
        0: np.array([-1, 0]),  # up
        1: np.array([1, 0]),   # down
        2: np.array([0, -1]),  # left
        3: np.array([0, 1])    # right
    }
    r, c = int(state_pos[0]), int(state_pos[1])
    dr, dc = moves[action]
    next_pos = np.array([r + dr, c + dc])
    # clip to grid
    next_pos = np.clip(next_pos, 0, grid_size - 1)

    # if next is obstacle -> reward -5, agent stays in place (matches env.step)
    for obs in env.obstacles:
        if np.array_equal(next_pos, obs):
            reward = -5.0
            terminated = False
            # agent does NOT move into obstacle (env leaves agent_pos unchanged)
            return np.array([r, c], dtype=np.int32), reward, terminated

    # if next is goal
    if np.array_equal(next_pos, env.goal_pos):
        reward = 10.0
        terminated = True
        return next_pos.astype(np.int32), reward, terminated

    # normal moved cell
    reward = -1.0
    terminated = False
    return next_pos.astype(np.int32), reward, terminated

# ----------------------------
# Value Iteration
# ----------------------------
def value_iteration(env, gamma=0.99, theta=1e-4, max_iters=10000):
    n = env.grid_size * env.grid_size
    V = np.zeros(n)  # value table
    policy = np.zeros(n, dtype=np.int32)

    for it in range(max_iters):
        delta = 0.0
        for s in range(n):
            pos = index_to_pos(s, env.grid_size)

            # skip if pos is goal? we still compute but it's fine
            action_values = []
            for a in range(env.action_space.n):
                next_pos, reward, terminated = simulate_step(env, pos, a)
                s_next = pos_to_index(next_pos, env.grid_size)
                # if terminated (goal), future value = 0 because episode ends
                future = 0.0 if terminated else V[s_next]
                q = reward + gamma * future
                action_values.append(q)

            new_v = float(np.max(action_values))
            delta = max(delta, abs(V[s] - new_v))
            V[s] = new_v

        if delta < theta:
            print(f"[ValueIteration] converged after {it+1} iterations (delta={delta:.6f})")
            break
    else:
        print(f"[ValueIteration] stopped after max_iters={max_iters} (delta={delta:.6f})")

    # extract greedy policy
    for s in range(n):
        pos = index_to_pos(s, env.grid_size)
        q_list = []
        for a in range(env.action_space.n):
            next_pos, reward, terminated = simulate_step(env, pos, a)
            s_next = pos_to_index(next_pos, env.grid_size)
            future = 0.0 if terminated else V[s_next]
            q = reward + gamma * future
            q_list.append(q)
        policy[s] = int(np.argmax(q_list))

    return V, policy

# ----------------------------
# Plot utilities
# ----------------------------
def plot_value_and_policy(env, V, policy, title="Value & Policy"):
    gs = env.grid_size
    Vgrid = V.reshape((gs, gs))
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(Vgrid, origin='upper', interpolation='nearest', cmap='viridis')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)

    # draw grid lines
    ax.set_xticks(np.arange(gs+1)-0.5)
    ax.set_yticks(np.arange(gs+1)-0.5)
    ax.grid(color='gray')

    # mark obstacles and goal and agent start
    for obs in env.obstacles:
        r, c = int(obs[0]), int(obs[1])
        ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='black'))
    gr, gc = int(env.goal_pos[0]), int(env.goal_pos[1])
    ax.add_patch(plt.Rectangle((gc-0.5, gr-0.5), 1, 1, color='green', alpha=0.8))
    # optional: mark start (0,0)
    sr, sc = 0, 0
    ax.add_patch(plt.Rectangle((sc-0.5, sr-0.5), 1, 1, edgecolor='blue', facecolor='none', linewidth=2))

    # draw policy arrows
    for r in range(gs):
        for c in range(gs):
            if any((r == int(obs[0]) and c == int(obs[1])) for obs in env.obstacles):
                continue
            if r == gr and c == gc:
                continue
            s = pos_to_index((r,c), gs)
            a = policy[s]
            dx, dy = 0.0, 0.0
            # recall env actions: 0:up,1:down,2:left,3:right
            if a == 0:
                dx, dy = 0, -0.3
            elif a == 1:
                dx, dy = 0, 0.3
            elif a == 2:
                dx, dy = -0.3, 0
            elif a == 3:
                dx, dy = 0.3, 0
            ax.arrow(c, r, dx, dy, head_width=0.12, head_length=0.12, fc='white', ec='white')

    ax.set_xlim(-0.5, gs-0.5)
    ax.set_ylim(gs-0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

# ----------------------------
# Policy-driven agent runner
# ----------------------------
def run_policy_episode(env, policy, max_steps=100, render=True, sleep=0.3):
    state, _ = env.reset()
    env.agent_pos = state.copy()
    traj = [state.copy()]
    for t in range(max_steps):
        s_idx = pos_to_index(state, env.grid_size)
        a = int(policy[s_idx])
        state, reward, terminated, truncated, info = env.step(a)
        traj.append(state.copy())
        if render:
            env.render()
            time.sleep(sleep)
        if terminated:
            print(f"Episode finished in {t+1} steps, reward={reward}")
            break
    else:
        print("Episode reached max steps without reaching terminal state.")
    return np.array(traj)

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    env = GridWorldEnv(render_mode="human")
    print("Running Value Iteration...")
    V, policy = value_iteration(env, gamma=0.99, theta=1e-5)
    print("Value function (reshaped):")
    print(V.reshape((env.grid_size, env.grid_size)))
    plot_value_and_policy(env, V, policy, title="Value Function & Greedy Policy")

    print("Now running an episode following the learned policy...")
    traj = run_policy_episode(env, policy, max_steps=50, render=True, sleep=0.15)

    # visualize trajectory on top of value heatmap
    fig, ax = plt.subplots(figsize=(6,6))
    gs = env.grid_size
    im = ax.imshow(V.reshape((gs, gs)), origin='upper', cmap='viridis')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # draw obstacles + goal
    for obs in env.obstacles:
        r, c = int(obs[0]), int(obs[1])
        ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='black'))
    gr, gc = int(env.goal_pos[0]), int(env.goal_pos[1])
    ax.add_patch(plt.Rectangle((gc-0.5, gr-0.5), 1, 1, color='green', alpha=0.8))
    # plot trajectory
    if len(traj) > 0:
        ax.plot(traj[:,1], traj[:,0], '-o', color='red')
    ax.set_xlim(-0.5, gs-0.5)
    ax.set_ylim(gs-0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title("Trajectory following learned policy")
    plt.show()
