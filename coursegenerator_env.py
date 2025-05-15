import gym
from gym import spaces
import numpy as np

class CourseGeneratorEnv(gym.Env):
    def __init__(self):
        super(CourseGeneratorEnv, self).__init__()
        # Define action and observation space
        # Actions: 0 = add lesson, 1 = add quiz, 2 = add assignment,
        #          3 = add feedback session, 4 = finalize course
        self.action_space = spaces.Discrete(5)

        # Observations: [num_lessons, num_quizzes, total_difficulty, total_duration, engagement_score]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0]),  # Minimum values for each state
            high=np.array([50, 20, 100, 500, 100]),  # Maximum values for each state
            dtype=np.float32
        )

        # Initial state
        self.state = np.array([0, 0, 0, 0, 50])  # Example initial values
        self.done = False

    def step(self, action):
        # Apply action and update state
        if action == 0:  # Add lesson
            self.state[0] += 1  # Increase number of lessons
            self.state[2] += 2  # Increase difficulty
            self.state[3] += 30  # Increase duration
            self.state[4] += 5  # Increase engagement
            reward = 5
        elif action == 1:  # Add quiz
            self.state[1] += 1  # Increase number of quizzes
            self.state[2] += 3  # Increase difficulty
            self.state[3] += 20  # Increase duration
            self.state[4] += 10  # Increase engagement
            reward = 7
        elif action == 2:  # Add assignment
            self.state[2] += 5  # Increase difficulty
            self.state[3] += 40  # Increase duration
            self.state[4] += 8  # Increase engagement
            reward = 6
        elif action == 3:  # Add feedback session
            self.state[4] += 15  # Increase engagement
            reward = 4
        elif action == 4:  # Finalize course
            self.done = True
            reward = 10 if self.state[0] >= 5 and self.state[1] >= 2 else -10  # Bonus for a balanced course
        else:
            reward = -1  # Invalid action

        # Cap state values
        self.state[2] = min(self.state[2], 100)  # Cap difficulty at 100
        self.state[3] = min(self.state[3], 500)  # Cap duration at 500
        self.state[4] = min(self.state[4], 100)  # Cap engagement at 100

        # Check if the course is finalized
        if self.done:
            reward += 20  # Additional reward for completing the course

        return self.state, reward, self.done, {}

    def reset(self):
        # Reset state
        self.state = np.array([0, 0, 0, 0, 50])  # Reset to initial values
        self.done = False
        return self.state

    def render(self, mode="human"):
        print(f"State: {self.state}")

if __name__ == "__main__":
    env = CourseGeneratorEnv()
    obs = env.reset()
    print("Initial State:", obs)

    for _ in range(20):  # Run for 20 steps
        action = env.action_space.sample()  # Take a random action
        obs, reward, done, info = env.step(action)
        env.render()  # Print the current state
        print(f"Action: {action}, Reward: {reward}, Done: {done}")
        if done:
            print("Course finalized!")
            break