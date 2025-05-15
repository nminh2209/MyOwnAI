from admission_consulting_env import AdmissionConsultingEnv

def print_action_menu():
    print("\nAvailable actions:")
    print("0: Add profile assessment")
    print("1: Add school selection")
    print("2: Add essay review")
    print("3: Add interview prep")
    print("4: Add application strategy")
    print("5: Finalize plan")
    print("q: Quit testing")

def main():
    env = AdmissionConsultingEnv()
    obs = env.reset()
    print("Initial State:")
    env.render()
    
    while True:
        print_action_menu()
        choice = input("Enter action (0-5, or q to quit): ")
        
        if choice.lower() == 'q':
            break
            
        try:
            action = int(choice)
            if action not in range(6):
                print("Invalid action! Please choose 0-5.")
                continue
                
            obs, reward, done, _ = env.step(action)
            print("\nCurrent State:")
            env.render()
            print(f"Reward: {reward}, Done: {done}")
            
            if done:
                print("\nPlan finalized! Starting a new plan.")
                obs = env.reset()
        except ValueError:
            print("Invalid input! Please enter a number or 'q'.")

if __name__ == "__main__":
    main()
