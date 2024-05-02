import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict(): 
    float_features = [eval(x) for x in request.form.values()]
    l = float_features[-1]
    float_features.pop(-1)
    for i in l:
        float_features.append(i)
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text = "The customer probably will be  {}".format(prediction[0]))
if __name__ == "__main__":
    flask_app.run(debug=True)

