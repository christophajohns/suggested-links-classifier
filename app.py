from flask import Flask, request
from utils import valid_data, qualifications

app = Flask(__name__)


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
        return qualifications(request.json)
    else:
        return {"qualifications": []}
