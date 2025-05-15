from stable_baselines3 import PPO
from coursegenerator_env import CourseGeneratorEnv  # Import the updated environment

# Load the environment and trained model
env = CourseGeneratorEnv()
model = PPO.load("course_generator_agent")  # Load the trained course generator model

# Test the agent
obs = env.reset()
for _ in range(20):  # Test for 20 steps
    action, _states = model.predict(obs)  # Agent decides the action
    obs, reward, done, info = env.step(action)  # Take the action
    env.render()  # Display the current state
    print(f"Action: {action}, Reward: {reward}, Done: {done}")
    if done:
        print("Course finalized!")
        break