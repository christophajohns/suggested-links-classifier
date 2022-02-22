from sklearn.linear_model import SGDClassifier
import numpy as np
from joblib import dump

from utils import feature_vector

initial_training_data = [
    {
        "source": {
            "id": "23:10",
            "color": {
                "r": 0.17,
                "g": 0.61,
                "b": 0.86,
            },
            "characters": "More details",
        },
        "target": {
            "id": "23:12",
            "topics": [
                "design",
                "technology",
                "shape",
                "detail",
                "classic",
            ],
        },
        "is_link": True,
    },
    {
        "source": {
            "id": "23:11",
            "color": {
                "r": 0,
                "g": 0,
                "b": 0,
            },
            "characters": "Shopping Bag",
        },
        "target": {
            "id": "23:12",
            "topics": [
                "total",
                "price",
                "checkout",
                "order",
                "shopping",
            ],
        },
        "is_link": True,
    },
    {
        "source": {
            "id": "23:11",
            "color": {
                "r": 0.05,
                "g": 0.05,
                "b": 0.05,
            },
            "characters": "Development",
        },
        "target": {
            "id": "23:12",
            "topics": [
                "history",
                "about",
                "team",
                "contact",
                "headquarters",
            ],
        },
        "is_link": False,
    },
    {
        "source": {
            "id": "23:13",
            "color": {
                "r": 0.51,
                "g": 0.51,
                "b": 0.51,
            },
            "characters": "Lorem ipsum dolor sit amet",
        },
        "target": {
            "id": "23:14",
            "topics": [
                "home",
                "cloud",
                "technique",
                "painting",
                "house",
            ],
        },
        "is_link": False,
    },
]


def get_training_and_target(initial_training_data):
    X = []
    Y = []
    for potential_link in initial_training_data:
        is_link = potential_link["is_link"]
        source = potential_link["source"]
        target = potential_link["target"]
        input_features = feature_vector(source, target)
        X.append(input_features)
        Y.append(int(is_link))
    return np.array(X), np.array(Y)


X, Y = get_training_and_target(initial_training_data)

clf = SGDClassifier(loss="log")
clf.fit(X, Y)

dump(clf, "classifier.joblib")
