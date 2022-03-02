from sklearn.linear_model import SGDClassifier
import numpy as np
from joblib import dump
import json
from tqdm import tqdm

from utils import feature_vector

# training_data_path = "sample_training_data.json"
training_data_path = "../rico-preprocessing/data/links.json"

with open(training_data_path) as training_data_json:
    initial_training_data = json.load(training_data_json)["links"]

    def get_training_and_target(initial_training_data):
        X = []
        Y = []
        for data_point in tqdm(initial_training_data):
            is_link = data_point["isLink"]
            source = data_point["link"]["source"]
            target = data_point["link"]["target"]
            context = data_point["link"]["context"]
            input_features = feature_vector(source, target, context)
            X.append(input_features)
            Y.append(int(is_link))
        return np.array(X), np.array(Y)

    X, Y = get_training_and_target(initial_training_data)

    with open('training.npy', 'wb') as f:
        np.save(f, X)
        np.save(f, Y)

    clf = SGDClassifier(loss="log", class_weight="balanced", verbose=1)
    clf.fit(X, Y)

    dump(clf, "classifiers/static.joblib")
