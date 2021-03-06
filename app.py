from flask import Flask, request, abort
from sklearn.exceptions import NotFittedError
from utils import validate_data, qualifications, update_classifier, create_classifier
from joblib import load
from os.path import exists

app = Flask(__name__)
static_classifier = load("classifiers/static.joblib")
static_clickable_classifier = load("classifiers/clickable.joblib")


@app.route("/")
def index():
    """Returns the version number of the application.

    :return: Application version
    :rtype: string
    """
    return "Suggested Link Classifier App v0.0.1"


@app.route("/qualifications", methods=["POST"])
def get_qualifications():
    try:
        validate_data(request.json)
        return qualifications(
            request.json, static_classifier, static_clickable_classifier
        )
    except Exception as error:
        print(error)
        abort(400)


@app.route("/model/<model_id>/qualifications", methods=["POST"])
def get_qualifications_from_interactive_model(model_id):
    classifier_path = f"classifiers/{model_id}.joblib"
    clickable_classifier_path = f"classifiers/{model_id}_clickable.joblib"
    try:
        validate_data(request.json)
        if exists(classifier_path) and exists(clickable_classifier_path):
            interactive_classifier = load(classifier_path)
            interactive_clickable_classifier = load(clickable_classifier_path)
            try:
                return qualifications(
                    request.json,
                    interactive_classifier,
                    interactive_clickable_classifier,
                )
            except NotFittedError as e:
                return qualifications(
                    request.json, static_classifier, static_clickable_classifier
                )
        create_classifier(model_id)
        return qualifications(
            request.json, static_classifier, static_clickable_classifier
        )
    except Exception as error:
        print(error)
        abort(400)


@app.route("/model/<model_id>/update", methods=["POST"])
def update_model(model_id):
    classifier_path = f"classifiers/{model_id}.joblib"
    if exists(classifier_path):
        update_classifier(request.json, model_id)
        return {"message": "model updated"}
    create_classifier(model_id)
    return {"message": "model created"}
