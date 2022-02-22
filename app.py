from flask import Flask, request

app = Flask(__name__)


@app.route("/")
def index():
    """Returns the version number of the application.

    :return: Application version
    :rtype: string
    """
    return "Suggested Link Classifier App v0.0.1"
