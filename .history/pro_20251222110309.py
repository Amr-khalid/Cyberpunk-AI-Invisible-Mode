import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import json
import random

app = Flask(__name__)
CORS(app)
url = "data.json"

def getData():
    if os.path.exists(url):
        try:
            with open(url, 'r') as f:
                data = json.load(f)
                if "userSteps" not in data: data['userSteps'] = []
                if "aiSteps" not in data: data['aiSteps'] = []
                if "lastMove" not in data: data['lastMove'] = None
                if "stats" not in data: data['stats'] = {"wins": 0, "losses": 0, "ties": 0}
                return data
        except:
            pass
    return {
        "stats": {"wins": 0, "losses": 0, "ties": 0},
        "userSteps": [],
        "aiSteps": [],
        "lastMove": None
    }

def save(data):
    with open(url, 'w') as f:
        json.dump(data, f)

option = {0: "rock", 1: "paper", 2: "scissors"}

@app.route('/play', methods=['POST'])
def play():
    data = getData()
    req = request.json
    if not req:
        return jsonify({"error": "no req"})

    user_move = int(req["move"])

    stats = data['stats']
    userSteps = data['userSteps']
    aiSteps = data['aiSteps']
    lastMove = data['lastMove']

    aiMove = random.randint(0, 2)
    mode = "random"
#[1] 0=>
    if len(userSteps) > 5:
        try:
            model = DecisionTreeClassifier()
            X = np.array(userSteps).reshape(-1, 1)
            y = np.array(aiSteps)
            model.fit(X, y)
            if lastMove is not None:
                predicted_user_move = model.predict([[lastMove]])[0]
                aiMove = int((predicted_user_move + 1) % 3)
                mode = "ai"
        except Exception as e:
            print("error:", e)

    if user_move == aiMove:
        result = "tie"
        stats["ties"] += 1
    elif (user_move == 0 and aiMove == 2) or \
         (user_move == 1 and aiMove == 0) or \
         (user_move == 2 and aiMove == 1):
        result = "win"
        stats["wins"] += 1
    else:
        result = "loss"
        stats["losses"] += 1

    if lastMove is not None:
        userSteps.append(lastMove)
        aiSteps.append(user_move)

    data["stats"] = stats
    data["userSteps"] = userSteps
    data["aiSteps"] = aiSteps
    data["lastMove"] = user_move

    save(data)

    return jsonify({
        "user_step": option[user_move],
        "ai_move": option[aiMove],
        "ai_move_int": aiMove,
        "result": result,
        "mode": mode,
        "stats": stats
    })

@app.route("/reset")
def reset():
    resetdata = {
        "userSteps": [],
        "aiSteps": [],
        "stats": {"wins": 0, "losses": 0, "ties": 0},
        "lastMove": None
    }
    save(resetdata)
    return jsonify({"status": "reset"})

@app.route('/', methods=['GET'])
def index():
    data = getData()
    return jsonify({"stats": data["stats"]})

if __name__ == "__main__":
    app.run(debug=True , port=5001)
