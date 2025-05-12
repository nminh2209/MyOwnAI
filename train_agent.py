# filepath: c:\Users\ADMIN\Documents\GitHub\MyOwnAI\train_agent.py
import gym
from stable_baselines3 import PPO
from mentoring_env import MentoringEnv

# Create the environment
env = MentoringEnv()

# Initialize the PPO agent
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0001, n_steps=2048, batch_size=64)

# Train the agent
model.learn(total_timesteps=100000)

# Save the trained model
model.save("mentoring_agent")