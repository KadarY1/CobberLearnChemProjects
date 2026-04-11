"""
FrozenLake Q-Learning Agent
--------------------------------
This script:
1. Creates FrozenLake-v1 environment
2. Initializes a Q-table
3. Trains using Q-learning
4. Evaluates performance
5. Visualizes learning progress
6. Displays policy as arrows
"""

# -------------------------------
# Imports
# -------------------------------
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Create Environment
# -------------------------------
env = gym.make("FrozenLake-v1", is_slippery=True)

# Print environment info
print("\n--- Environment Info ---")
print("Observation Space (States):", env.observation_space.n)
print("Action Space (Actions):", env.action_space.n)
print("Actions: 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP")

# -------------------------------
# Step 2: Initialize Q-table
# -------------------------------
state_size = env.observation_space.n   # 16 states
action_size = env.action_space.n       # 4 actions

# Initialize Q-table with zeros
Q = np.zeros((state_size, action_size))

# -------------------------------
# Step 3: Hyperparameters
# -------------------------------
learning_rate = 0.8
gamma = 0.95           # Discount factor
epsilon = 1.0          # Exploration rate
epsilon_decay = 0.995
epsilon_min = 0.01

episodes = 5000
max_steps = 100

rewards_per_episode = []

# -------------------------------
# Step 4: Training Loop
# -------------------------------
for episode in range(episodes):
    state, _ = env.reset()
    total_reward = 0

    for step in range(max_steps):
        # Exploration vs Exploitation
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        # Take action
        new_state, reward, terminated, truncated, _ = env.step(action)

        # Q-learning update rule
        Q[state, action] = Q[state, action] + learning_rate * (
            reward + gamma * np.max(Q[new_state, :]) - Q[state, action]
        )

        state = new_state
        total_reward += reward

        if terminated or truncated:
            break

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    rewards_per_episode.append(total_reward)

# -------------------------------
# Step 5: Print Final Q-table
# -------------------------------
print("\n--- Final Q-table ---")
print(Q)

# -------------------------------
# Step 6: Policy Visualization
# -------------------------------
def display_policy(Q):
    """
    Converts Q-table into directional arrows
    """
    arrows = ["←", "↓", "→", "↑"]
    policy_grid = []

    for state in range(state_size):
        best_action = np.argmax(Q[state])
        policy_grid.append(arrows[best_action])

    # Convert to 4x4 grid
    policy_grid = np.array(policy_grid).reshape(4, 4)

    print("\n--- Learned Policy ---")
    for row in policy_grid:
        print(" ".join(row))

display_policy(Q)

# -------------------------------
# Step 7: Test the Agent
# -------------------------------
test_episodes = 100
successes = 0

for _ in range(test_episodes):
    state, _ = env.reset()

    for _ in range(max_steps):
        action = np.argmax(Q[state])  # NO exploration
        state, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            if reward == 1:
                successes += 1
            break

success_rate = successes / test_episodes * 100

print(f"\n--- Test Results ---")
print(f"Success Rate: {success_rate:.2f}%")

if success_rate > 70:
    print("Good learning achieved!")
elif success_rate < 30:
    print("Agent needs more training.")
else:
    print("Moderate performance.")

# -------------------------------
# Step 8: Plot Learning Curve
# -------------------------------
# Smooth rewards for better visualization
window = 100
smoothed_rewards = np.convolve(rewards_per_episode, np.ones(window)/window, mode='valid')

plt.figure()
plt.plot(smoothed_rewards)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Learning Curve")
plt.savefig("learning_curve.png")
plt.close()

print("Saved visualization: learning_curve.png")

# -------------------------------
# Step 9: Hyperparameter Experimentation
# -------------------------------
print("\n--- Hyperparameter Suggestions ---")
print("Try changing:")
print("- learning_rate (e.g., 0.1, 0.5, 0.9)")
print("- gamma (e.g., 0.8, 0.95, 0.99)")
print("- epsilon_decay (e.g., 0.99, 0.995, 0.999)")
print("Observe how success rate and learning curve change.")