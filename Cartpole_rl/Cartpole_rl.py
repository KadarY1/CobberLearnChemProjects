import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import imageio

# ---------------------------------------------------
# Create CartPole Environment
# We will save frames manually using imageio
# instead of using RecordVideo + moviepy
# ---------------------------------------------------

env = gym.make("CartPole-v1", render_mode="rgb_array")

# ---------------------------------------------------
# Random Policy
# ---------------------------------------------------

def random_policy(state):
    return random.choice([0, 1])


# ---------------------------------------------------
# Rule-Based Policy
# ---------------------------------------------------

def rule_based_policy(state):
    pole_angle = state[2]

    if pole_angle < 0:
        return 0  # move left
    else:
        return 1  # move right


# ---------------------------------------------------
# Run Episode + Save Video Frames
# ---------------------------------------------------

def run_episode_with_video(policy_function, save_video=False):
    state, info = env.reset()

    done = False
    total_reward = 0
    steps = 0
    frames = []

    while not done:
        # Capture frame
        if save_video:
            frame = env.render()
            frames.append(frame)

        action = policy_function(state)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        state = next_state
        total_reward += reward
        steps += 1

    # Save video manually using imageio
    if save_video and len(frames) > 0:
        imageio.mimsave("cartpole_simulation.gif", frames, fps=30)
        print("Video saved as: cartpole_simulation.gif")

    return total_reward, steps


# ---------------------------------------------------
# Run Example
# Save one GIF of rule-based policy
# ---------------------------------------------------

reward, steps = run_episode_with_video(
    rule_based_policy,
    save_video=True
)

print(f"Total Reward: {reward}")
print(f"Steps Survived: {steps}")

env.close()