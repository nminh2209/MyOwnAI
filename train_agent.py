from stable_baselines3 import PPO
from coursegenerator_env import CourseGeneratorEnv  # Import the updated environment

# Create the environment
env = CourseGeneratorEnv()

# Initialize the PPO agent
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0001, n_steps=2048, batch_size=64)

# Train the agent
model.learn(total_timesteps=100000)

# Save the trained model
model.save("course_generator_agent")