from stable_baselines3 import PPO
from mentoring_env import MentoringEnv

# Load the environment and trained model
env = MentoringEnv()
model = PPO.load("mentoring_agent")

# Test the agent
obs = env.reset()
for _ in range(100):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        print("Course completed!")
        break