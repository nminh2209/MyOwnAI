import gym
from gym import spaces
import numpy as np
import os
import zipfile
import json
from datetime import datetime

class AdmissionConsultingEnv(gym.Env):
    def __init__(self):
        super(AdmissionConsultingEnv, self).__init__()
        # Define action and observation space
        # Actions: 0 = add profile assessment, 1 = add school selection, 2 = add essay review,
        #          3 = add interview prep, 4 = add application strategy, 5 = finalize plan
        self.action_space = spaces.Discrete(6)

        # Observations: [profile_assessments, school_selections, essay_reviews, interview_preps, 
        #                strategy_sessions, completeness_score, personalization_score]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0]),  # Minimum values for each state
            high=np.array([5, 10, 20, 10, 5, 100, 100]),  # Maximum values for each state
            dtype=np.float32
        )

        # Initial state
        self.state = np.array([0, 0, 0, 0, 0, 20, 30])  # Example initial values
        self.done = False

        # Consulting plan to store generated content
        self.consulting_plan = []

    def step(self, action):
        # Apply action and update state
        if action == 0:  # Add profile assessment
            self.state[0] += 1  # Increase number of profile assessments
            self.state[5] += 15  # Increase completeness
            self.state[6] += 10  # Increase personalization
            self.consulting_plan.append(f"Profile Assessment {self.state[0]}: Strengths & Weaknesses Analysis")
            reward = 8
        elif action == 1:  # Add school selection
            self.state[1] += 1  # Increase number of school selections
            self.state[5] += 10  # Increase completeness
            self.state[6] += 15  # Increase personalization
            self.consulting_plan.append(f"School Selection {self.state[1]}: Target Schools Analysis")
            reward = 7
        elif action == 2:  # Add essay review
            self.state[2] += 1  # Increase number of essay reviews
            self.state[5] += 12  # Increase completeness
            self.state[6] += 20  # Increase personalization
            self.consulting_plan.append(f"Essay Review {self.state[2]}: Personal Statement Feedback")
            reward = 10
        elif action == 3:  # Add interview prep
            self.state[3] += 1  # Increase number of interview preps
            self.state[5] += 8  # Increase completeness
            self.state[6] += 12  # Increase personalization
            self.consulting_plan.append(f"Interview Preparation {self.state[3]}: Mock Interview Session")
            reward = 6
        elif action == 4:  # Add application strategy
            self.state[4] += 1  # Increase number of strategy sessions
            self.state[5] += 18  # Increase completeness
            self.state[6] += 18  # Increase personalization
            self.consulting_plan.append(f"Application Strategy {self.state[4]}: Timeline and Approach Planning")
            reward = 9
        elif action == 5:  # Finalize plan
            self.done = True
            # Bonus for a balanced plan with at least one of each component
            reward = 20 if all(self.state[:5] >= 1) else -5
        else:
            reward = -1  # Invalid action

        # Cap state values
        self.state[5] = min(self.state[5], 100)  # Cap completeness at 100
        self.state[6] = min(self.state[6], 100)  # Cap personalization at 100

        # Check if the plan is finalized
        if self.done:
            # Additional reward based on plan quality
            quality_bonus = min(15, sum(self.state[:5]) * 2)
            reward += quality_bonus

        return self.state, reward, self.done, {}

    def reset(self):
        # Reset state
        self.state = np.array([0, 0, 0, 0, 0, 20, 30])  # Reset to initial values
        self.done = False
        self.consulting_plan = []  # Clear the consulting plan
        return self.state

    def render(self, mode="human"):
        print(f"State: {self.state}")
        print(f"Profile Assessments: {self.state[0]}, School Selections: {self.state[1]}, Essay Reviews: {self.state[2]}")
        print(f"Interview Preps: {self.state[3]}, Strategy Sessions: {self.state[4]}")
        print(f"Completeness: {self.state[5]}, Personalization: {self.state[6]}")
        
        if self.done:
            print("\nGenerated Admission Consulting Plan:")
            for i, item in enumerate(self.consulting_plan, 1):
                print(f"{i}. {item}")
    
    def export_plan_as_zip(self, student_name="student", output_dir="plans"):
        """
        Export the consulting plan as a zip file
        
        Args:
            student_name: Name of the student for the filename
            output_dir: Directory to save the zip file
        
        Returns:
            Path to the created zip file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"{output_dir}/{student_name}_{timestamp}_plan.zip"
        
        # Create a zip file
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            # Add the plan as a text file
            plan_text = "\n".join([f"{i+1}. {item}" for i, item in enumerate(self.consulting_plan)])
            zipf.writestr("consulting_plan.txt", plan_text)
            
            # Add the plan as JSON for programmatic access
            plan_data = {
                "student_name": student_name,
                "timestamp": timestamp,
                "state": self.state.tolist(),
                "plan_items": self.consulting_plan
            }
            zipf.writestr("plan_data.json", json.dumps(plan_data, indent=2))
            
            # Add a summary file
            summary = f"""
Admission Consulting Plan Summary
--------------------------------
Student: {student_name}
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Components:
- Profile Assessments: {self.state[0]}
- School Selections: {self.state[1]}
- Essay Reviews: {self.state[2]}
- Interview Preps: {self.state[3]}
- Strategy Sessions: {self.state[4]}

Metrics:
- Completeness Score: {self.state[5]:.1f}
- Personalization Score: {self.state[6]:.1f}

Total Items: {len(self.consulting_plan)}
"""
            zipf.writestr("summary.txt", summary)
        
        print(f"Plan exported to {zip_filename}")
        return zip_filename

if __name__ == "__main__":
    env = AdmissionConsultingEnv()
    obs = env.reset()
    print("Initial State:", obs)

    for episode in range(3):
        obs = env.reset()
        print(f"\n--- Episode {episode+1} ---")
        
        for step in range(15):  # Run for 15 steps
            action = env.action_space.sample()  # Take a random action
            obs, reward, done, info = env.step(action)
            print(f"\nStep {step+1}")
            env.render()  # Print the current state
            print(f"Action: {action}, Reward: {reward}, Done: {done}")
            
            if done:
                print("Consulting plan finalized!")
                # Export the plan as a zip file
                env.export_plan_as_zip(f"student_episode_{episode+1}")
                break
