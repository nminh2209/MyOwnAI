from stable_baselines3 import PPO
from admission_consulting_env import AdmissionConsultingEnv

def generate_personalized_plan(student_profile):
    """
    Generate a personalized admission consulting plan based on student profile
    
    Args:
        student_profile: Dictionary containing student information
    
    Returns:
        List of consulting plan elements
    """
    # Load the trained model
    model = PPO.load("admission_consulting_agent")
    
    # Create environment
    env = AdmissionConsultingEnv()
    obs = env.reset()
    
    # Adjust initial state based on student profile
    # For example, if student needs more essay help:
    if student_profile.get("essay_help_needed", False):
        # Bias the agent toward generating more essay reviews
        pass
    
    # Generate plan
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
    
    return env.consulting_plan

# Example usage
if __name__ == "__main__":
    student = {
        "name": "Alex Johnson",
        "target_degree": "MBA",
        "gpa": 3.7,
        "work_experience": 3,
        "essay_help_needed": True,
        "interview_anxiety": True
    }
    
    plan = generate_personalized_plan(student)
    
    print(f"Personalized Admission Consulting Plan for {student['name']}:")
    for i, item in enumerate(plan, 1):
        print(f"{i}. {item}")
