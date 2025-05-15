from generate_consulting_plan import generate_personalized_plan

# Test with different student profiles
students = [
    {
        "name": "Alex Johnson",
        "target_degree": "MBA",
        "gpa": 3.7,
        "work_experience": 3,
        "essay_help_needed": True,
        "interview_anxiety": True
    },
    {
        "name": "Sarah Williams",
        "target_degree": "Computer Science PhD",
        "gpa": 3.9,
        "work_experience": 1,
        "essay_help_needed": False,
        "interview_anxiety": False
    }
]

for student in students:
    print(f"\n{'='*50}")
    print(f"Generating plan for {student['name']} ({student['target_degree']})")
    print(f"{'='*50}")
    
    plan = generate_personalized_plan(student)
    
    print(f"\nPersonalized Admission Consulting Plan:")
    for i, item in enumerate(plan, 1):
        print(f"{i}. {item}")
