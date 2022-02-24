from sklearn.linear_model import SGDClassifier
import numpy as np
from joblib import dump
import json

from utils import feature_vector

with open("sample_training_data.json") as training_data_json:
    initial_training_data = json.load(training_data_json)["data"]

    def get_training_and_target(initial_training_data):
        X = []
        Y = []
        for data_point in initial_training_data:
            is_link = data_point["is_link"]
            source = data_point["link"]["source"]
            target = data_point["link"]["target"]
            input_features = feature_vector(source, target)
            X.append(input_features)
            Y.append(int(is_link))
        return np.array(X), np.array(Y)

    X, Y = get_training_and_target(initial_training_data)

    clf = SGDClassifier(loss="log")
    clf.fit(X, Y)

    dump(clf, "classifiers/static.joblib")
