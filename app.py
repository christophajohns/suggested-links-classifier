from flask import Flask, request
from sklearn.exceptions import NotFittedError
from utils import valid_data, qualifications, update_classifier, create_classifier
from joblib import load
from os.path import exists

app = Flask(__name__)
static_classifier = load("classifiers/static.joblib")


@app.route("/")
def index():
    """Returns the version number of the application.

    :return: Application version
    :rtype: string
    """
    return "Suggested Link Classifier App v0.0.1"


@app.route("/qualifications", methods=["POST"])
def get_qualifications():
    if valid_data(request.json):
        return qualifications(request.json, static_classifier)
    else:
        return {"qualifications": []}


@app.route("/model/<model_id>/qualifications", methods=["POST"])
def get_qualifications_from_interactive_model(model_id):
    classifier_path = f"classifiers/{model_id}.joblib"
    if valid_data(request.json):
        if exists(classifier_path):
            interactive_classifier = load(classifier_path)
            try:
                return qualifications(request.json, interactive_classifier)
            except NotFittedError as e:
                return qualifications(request.json, static_classifier)
        create_classifier(model_id)
        return qualifications(request.json, static_classifier)
    else:
        return {"qualifications": []}


@app.route("/model/<model_id>/update", methods=["POST"])
def update_model(model_id):
    classifier_path = f"classifiers/{model_id}.joblib"
    if exists(classifier_path):
        update_classifier(request.json, model_id)
        return {"message": "model updated"}
    create_classifier(model_id)
    return {"message": "model created"}
