from flask import Flask, request
from utils import valid_data, qualifications
from joblib import load

app = Flask(__name__)
clf = load("classifier.joblib")


@app.route("/")
def index():
    """Returns the version number of the application.

    :return: Application version
    :rtype: string
    """
    return "Suggested Link Classifier App v0.0.1"


@app.route("/qualifications", methods=["POST"])
def links():
    if valid_data(request.json):
        return qualifications(request.json, clf)
    else:
        return {"qualifications": []}
