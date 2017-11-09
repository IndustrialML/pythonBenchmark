import numpy as np
from flask import Flask, request

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict_digit():
    request_data = request.get_json()
    image = np.array(request_data)

    pred = 0
    return "%s"%pred

@app.route("/example", methods=["GET"])
def predict_example():

    pred = 0
    return "%s"%pred

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000)
