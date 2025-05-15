from flask import Flask, request, jsonify, render_template
from coursegenerator_env import MentoringEnv
from stable_baselines3 import PPO

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = PPO.load("mentoring_agent")

# Initialize the environment
env = MentoringEnv()
obs = env.reset()

@app.route("/")
def home():
    # Render the main interface
    return render_template("index.html", state=obs.tolist())

@app.route("/step", methods=["POST"])
def step():
    global obs
    # Get the action from the user or the trained agent
    action = request.json.get("action", None)
    if action is None:
        # Let the trained agent decide the action
        action, _ = model.predict(obs)

    # Take the action in the environment
    obs, reward, done, info = env.step(action)

    # Return the updated state, reward, and done status
    return jsonify({
        "state": obs.tolist(),
        "reward": reward,
        "done": done
    })

@app.route("/reset", methods=["POST"])
def reset():
    global obs
    # Reset the environment
    obs = env.reset()
    return jsonify({"state": obs.tolist()})

if __name__ == "__main__":
    app.run(debug=True)