from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier
import numpy as np
from joblib import dump
import json
from tqdm import tqdm

# training_data_path = "sample_training_data.json"
training_data_path = "../Screen2Vec/simplifiedscreen2vec/data/links.json"

with open("training_screen2vec.npy", "rb") as f:
    X = np.load(f)
    Y = np.load(f)

with open(training_data_path) as training_data_json:
    initial_training_data = json.load(training_data_json)["links"]


def get_training_and_target(initial_training_data):
    X = []
    Y = []
    for data_point in tqdm(initial_training_data):
        is_link = data_point["isLink"]
        source = data_point["link"]["source"]
        target = data_point["link"]["target"]
        X.append(source + target)
        Y.append(int(is_link))
    return np.array(X), np.array(Y)


X, Y = get_training_and_target(initial_training_data)

# with open("training_screen2vec.npy", "wb") as f:
#     np.save(f, X)
#     np.save(f, Y)

# clf = make_pipeline(
#     StandardScaler(), SGDClassifier(loss="log", class_weight="balanced", verbose=1)
# )
# clf = RandomForestClassifier(random_state=42, n_jobs=2, verbose=1)
# clf = BalancedBaggingClassifier(
#     base_estimator=HistGradientBoostingClassifier(random_state=42),
#     n_estimators=10,
#     random_state=42,
#     n_jobs=2,
#     verbose=1,
# )
clf = BalancedRandomForestClassifier(random_state=42, n_jobs=2, verbose=1)
clf.fit(X, Y)

# dump(clf, "classifiers/static_screen2vec.joblib")
dump(clf, "classifiers/static.joblib")
