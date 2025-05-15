import matplotlib.pyplot as plt
import numpy as np
from admission_consulting_env import AdmissionConsultingEnv

# Run multiple episodes and collect data
env = AdmissionConsultingEnv()
episodes = 10
max_steps = 20
all_rewards = []
component_counts = []

for episode in range(episodes):
    obs = env.reset()
    episode_rewards = []
    
    for step in range(max_steps):
        action = env.action_space.sample()  # Random action
        obs, reward, done, _ = env.step(action)
        episode_rewards.append(reward)
        
        if done:
            break
    
    all_rewards.append(sum(episode_rewards))
    component_counts.append(env.state[:5].copy())  # First 5 elements are component counts

# Plot rewards
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.bar(range(episodes), all_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Rewards per Episode')

# Plot component distribution for the last episode
plt.subplot(1, 2, 2)
components = ['Profile', 'Schools', 'Essays', 'Interviews', 'Strategy']
plt.bar(components, component_counts[-1])
plt.ylabel('Count')
plt.title('Components in Final Episode')

plt.tight_layout()
plt.savefig('results_visualization.png')
plt.show()
