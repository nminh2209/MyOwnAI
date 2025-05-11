import gym
from stable_baselines3 import PPO
from mentoring_env import MentoringEnv

# Create the environment
env = MentoringEnv()

# Initialize the PPO agent
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=10000)

# Save the trained model
model.save("mentoring_agent")