
from flask import Flask, render_template, request, jsonify

from chat import get_response

app = Flask(__name__)

@app.get("/")
def index_get():
    return render_template("base.html")

@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    
    # Debugging output
    print("Incoming message:", text)
    
    # TODO: check if text is valid
    response = get_response(text, score_resp)
    
    # Debugging output
    print("Response:", response)
    
    # Add the score to a previous array
    score_resp.append(response[1][0])
    
    # Bot response
    response_text = response[0]
    message = {"answer": response_text}
    
    return jsonify(message)

if __name__ == "__main__":
    score_resp = []
    app.run(debug=True)
