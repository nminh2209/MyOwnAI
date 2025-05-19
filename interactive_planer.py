import os
import traceback
from stable_baselines3 import PPO
from admission_consulting_env import AdmissionConsultingEnv

def get_student_info():
    """Get student information from user input"""
    print("\n=== Student Information ===")
    name = input("Student name: ").strip().replace(" ", "_").lower()
    target_degree = input("Target degree (e.g., MBA, PhD): ").strip()
    
    gpa = 0
    while gpa <= 0 or gpa > 4.0:
        try:
            gpa = float(input("GPA (0-4.0): ").strip())
        except ValueError:
            print("Please enter a valid number.")
    
    work_exp = -1
    while work_exp < 0:
        try:
            work_exp = int(input("Years of work experience: ").strip())
        except ValueError:
            print("Please enter a valid number.")
    
    essay_help = input("Needs essay help? (y/n): ").strip().lower() == 'y'
    interview_help = input("Needs interview preparation? (y/n): ").strip().lower() == 'y'
    
    return {
        "name": name,
        "target_degree": target_degree,
        "gpa": gpa,
        "work_experience": work_exp,
        "essay_help_needed": essay_help,
        "interview_anxiety": interview_help
    }

def generate_plan_with_model(model, env, student=None):
    """Generate a plan using the trained model"""
    try:
        obs = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Debug information
            print(f"Current observation shape: {obs.shape}, type: {type(obs)}")
            print(f"Current observation values: {obs}")
            
            # Predict action
            action, _ = model.predict(obs, deterministic=True)
            print(f"Predicted action: {action}")
            
            # Take action
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        
        print(f"\nPlan generated with total reward: {total_reward:.2f}")
        return True
    except Exception as e:
        print(f"Error generating plan: {e}")
        print("Detailed error:")
        traceback.print_exc()
        return False

def generate_basic_plan(env):
    """Generate a basic plan without using a model"""
    print("Generating a basic plan...")
    # Add one of each component type, then finalize
    for action in [0, 1, 2, 3, 4, 5]:
        obs, reward, done, _ = env.step(action)
        print(f"Added component with action {action}, reward: {reward}")
        if done:
            break
    return True

def main():
    # Create environment
    env = AdmissionConsultingEnv()
    
    # Check if model exists
    model_path = "admission_consulting_agent"
    model_exists = os.path.exists(f"{model_path}.zip")
    
    if not model_exists:
        print(f"Model file '{model_path}.zip' not found.")
        train = input("Would you like to train the model now? (y/n): ").strip().lower()
        if train == 'y':
            print("Training model... (this may take a few minutes)")
            os.system("python train_admission_agent.py")
            model_exists = os.path.exists(f"{model_path}.zip")
        else:
            print("Will use basic plan generation instead.")
    
    # Load the model if it exists
    model = None
    if model_exists:
        try:
            print(f"Loading model from {model_path}.zip")
            model = PPO.load(model_path)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            model_exists = False
    
    while True:
        print("\n=== AI Admission Consulting Planner ===")
        print("1. Generate plan with student information")
        print("2. Generate quick plan")
        print("3. Exit")
        
        choice = input("Select an option (1-3): ").strip()
        
        if choice == '1':
            student = get_student_info()
            
            # Generate plan
            success = False
            if model_exists and model is not None:
                print("Generating plan using trained model...")
                success = generate_plan_with_model(model, env, student)
            
            if not success:
                print("Falling back to basic plan generation...")
                success = generate_basic_plan(env)
            
            # Display and save plan
            if success:
                print("\nGenerated Admission Consulting Plan:")
                env.render()
                
                save = input("Save this plan? (y/n): ").strip().lower()
                if save == 'y':
                    zip_path = env.export_plan_as_zip(student['name'])
                    print(f"Plan saved to {zip_path}")
        
        elif choice == '2':
            # Quick plan with default student
            success = False
            if model_exists and model is not None:
                print("Generating quick plan using trained model...")
                success = generate_plan_with_model(model, env)
            
            if not success:
                print("Falling back to basic plan generation...")
                success = generate_basic_plan(env)
            
            if success:
                print("\nGenerated Quick Admission Consulting Plan:")
                env.render()
                
                save = input("Save this plan? (y/n): ").strip().lower()
                if save == 'y':
                    name = input("Enter a name for this plan: ").strip().replace(" ", "_").lower()
                    zip_path = env.export_plan_as_zip(name)
                    print(f"Plan saved to {zip_path}")
        
        elif choice == '3':
            print("Exiting program. Goodbye!")
            break
        
        else:
            print("Invalid choice. Please select 1, 2, or 3.")

if __name__ == "__main__":
    main()
