import gym
from gym import spaces
import numpy as np

class MentoringEnv(gym.Env):
    def __init__(self):
        super(MentoringEnv, self).__init__()
        # Define action and observation space
        # Actions: 0 = suggest topic, 1 = assign quiz, 2 = provide feedback
        self.action_space = spaces.Discrete(3)
        # Observations: [completed_lessons, quiz_scores, motivation_level]
        self.observation_space = spaces.Box(low=0, high=100, shape=(3,), dtype=np.float32)
        self.state = np.array([0, 0, 100])  # Initial state
        self.done = False

    def step(self, action):
        # Apply action and update state
        if action == 0:  # Suggest topic
            self.state[0] += 10  # Increase completed lessons
            reward = 5
        elif action == 1:  # Assign quiz
            self.state[1] += 5  # Increase quiz scores
            reward = 1
        elif action == 2:  # Provide feedback
            self.state[2] += 10  # Increase motivation
            reward = 3
        else:
            reward = -1  # Invalid action
            
             # Cap motivation level at 100
        self.state[2] = min(self.state[2], 100)

        # Check if the course is completed
        if self.state[0] >= 100:
            self.done = True
            reward += 10  # Bonus for completing the course

        return self.state, reward, self.done, {}

    def reset(self):
        self.state = np.array([0, 0, 100])  # Reset state
        self.done = False
        return self.state

    def render(self, mode="human"):
        print(f"State: {self.state}")

# Test the environment
if __name__ == "__main__":
    env = MentoringEnv()
    obs = env.reset()
    print("Initial State:", obs)

    for _ in range(10):
        action = env.action_space.sample()  # Random action
        obs, reward, done, info = env.step(action)
        env.render()
        print(f"Action: {action}, Reward: {reward}, Done: {done}")
        if done:
            print("Course completed!")
            break