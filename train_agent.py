import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from admission_consulting_env import AdmissionConsultingEnv

# Create environment
env = AdmissionConsultingEnv()

# Initialize the agent
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)

# Train the agent
model.learn(total_timesteps=50000)

# Save the model
model.save("admission_consulting_agent")

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Generate a sample consulting plan
obs = env.reset()
done = False
total_reward = 0

print("\n--- Generated Admission Consulting Plan ---")
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    total_reward += reward
    
env.render()
print(f"Total reward: {total_reward:.2f}")
