from flask import Flask, request, jsonify, render_template
from model import get_conversation_response

app = Flask(__name__)
conversation_history = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    global conversation_history
    data = request.json
    user_input = data.get("message", "")  # Ensure the key exists

    if not user_input:
        return jsonify({"response": "Please provide a message."})

    response, conversation_history = get_conversation_response(user_input, conversation_history)

    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=False)
