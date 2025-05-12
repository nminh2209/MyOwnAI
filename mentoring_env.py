import gym
from gym import spaces
import numpy as np

class MentoringEnv(gym.Env):
    def __init__(self):
        super(MentoringEnv, self).__init__()
        # Define action and observation space
        # Actions: 0 = suggest topic, 1 = assign quiz, 2 = provide feedback,
        #          3 = assign homework, 4 = schedule meeting
        self.action_space = spaces.Discrete(5)

        # Observations: [completed_lessons, quiz_scores, motivation_level, time_spent, engagement_level, difficulty_level]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 1]),  # Minimum values for each state
            high=np.array([100, 100, 100, 100, 100, 10]),  # Maximum values for each state
            dtype=np.float32
        )

        # Initial state
        self.state = np.array([0, 0, 0, 0, 0, 0])  # Example initial values
        self.done = False

    def step(self, action):
        # Apply action and update state
        if action == 0:  # Suggest topic
            self.state[0] += 10  # Increase completed lessons
            reward = 5
        elif action == 1:  # Assign quiz
            self.state[1] += 5  # Increase quiz scores
            reward = 2
        elif action == 2:  # Provide feedback
            self.state[2] += 10  # Increase motivation
            reward = 3
        elif action == 3:  # Assign homework
            self.state[3] += 5  # Increase time spent
            reward = 4
        elif action == 4:  # Schedule meeting
            self.state[4] += 10  # Increase engagement level
            reward = 6
        else:
            reward = -1  # Invalid action

        # Cap state values
        self.state[2] = min(self.state[2], 100)  # Cap motivation at 100
        self.state[3] = min(self.state[3], 100)  # Cap time spent at 100
        self.state[4] = min(self.state[4], 100)  # Cap engagement level at 100

        # Check if the course is completed
        if self.state[0] >= 100:  # Completed lessons
            self.done = True
            reward += 10  # Bonus for completing the course
        else:
            self.done = False

        return self.state, reward, self.done, {}

    def reset(self):
        # Reset state
        self.state = np.array([0, 0, 100, 0, 50, 5])  # Reset to initial values
        self.done = False
        return self.state

    def render(self, mode="human"):
        print(f"State: {self.state}")

if __name__ == "__main__":
    env = MentoringEnv()
    obs = env.reset()
    print("Initial State:", obs)

    for _ in range(20):  # Run for 20 steps
        action = env.action_space.sample()  # Take a random action
        obs, reward, done, info = env.step(action)
        env.render()  # Print the current state
        print(f"Action: {action}, Reward: {reward}, Done: {done}")
        if done:
            print("Course completed!")
            break