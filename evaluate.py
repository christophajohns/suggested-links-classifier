import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="evaluation_logs.log",
    level=logging.INFO,
)
logging.info("Starting evaluation.")

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_validate
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
import numpy as np
import pandas as pd
from sklearn.naive_bayes import ComplementNB
from tqdm import tqdm
from pprint import pprint
from imblearn.pipeline import make_pipeline as make_pipeline_with_sampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier
from imblearn.metrics import sensitivity_score


with open("training_screen2vec.npy", "rb") as f:
    X = np.load(f)
    Y = np.load(f)

linear_scaler = StandardScaler()
nb_scaler = MinMaxScaler()

index = []
scores = {
    "Accuracy": [],
    "Balanced accuracy": [],
    "Sensitivity": [],
    "Precision": [],
    "Recall": [],
    "F1": [],
}
scoring = {
    "accuracy": make_scorer(accuracy_score),
    "balanced_accuracy": make_scorer(balanced_accuracy_score),
    "sensitivity": make_scorer(sensitivity_score),
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score),
    "f1": make_scorer(f1_score),
}

dummy_clf = DummyClassifier(strategy="most_frequent")
lr_clf = make_pipeline(linear_scaler, SGDClassifier(loss="log"))
rf_clf = RandomForestClassifier(random_state=42, n_jobs=2)
lr_balanced_clf = make_pipeline(
    linear_scaler, SGDClassifier(loss="log", class_weight="balanced")
)
rf_balanced_clf = RandomForestClassifier(
    random_state=42, n_jobs=2, class_weight="balanced"
)
nb_clf = make_pipeline(nb_scaler, ComplementNB())
lr_sampler_clf = make_pipeline_with_sampler(
    linear_scaler,
    RandomUnderSampler(random_state=42),
    SGDClassifier(loss="log"),
)
rf_sampler_clf = make_pipeline_with_sampler(
    RandomUnderSampler(random_state=42),
    RandomForestClassifier(random_state=42, n_jobs=2),
)
rf_imblearn_balanced_clf = BalancedRandomForestClassifier(random_state=42, n_jobs=2)
bag_clf = BalancedBaggingClassifier(
    base_estimator=HistGradientBoostingClassifier(random_state=42),
    n_estimators=10,
    random_state=42,
    n_jobs=2,
)
lr_smote_clf = make_pipeline_with_sampler(
    linear_scaler,
    SMOTE(random_state=42),
    SGDClassifier(loss="log"),
)
rf_smote_clf = make_pipeline_with_sampler(
    linear_scaler,
    SMOTE(random_state=42),
    RandomForestClassifier(random_state=42, n_jobs=2),
)
lr_adasyn_clf = make_pipeline_with_sampler(
    linear_scaler,
    ADASYN(random_state=42),
    SGDClassifier(loss="log"),
)
rf_adasyn_clf = make_pipeline_with_sampler(
    linear_scaler,
    ADASYN(random_state=42),
    RandomForestClassifier(random_state=42, n_jobs=2),
)

classifiers = [
    {"classifier": dummy_clf},
    {"classifier": lr_clf, "name": "SGDClassifier"},
    {"classifier": rf_clf},
    {"classifier": lr_balanced_clf, "name": "SGDClassifier (balanced)"},
    {"classifier": rf_balanced_clf, "name": "RandomForestClassifier (balanced)"},
    {"classifier": nb_clf, "name": "ComplementNB"},
    {"classifier": lr_sampler_clf, "name": "SGDClassifier (under-sampling)"},
    {"classifier": rf_sampler_clf, "name": "RandomForestClassifier (under-sampling)"},
    {"classifier": rf_imblearn_balanced_clf},
    {"classifier": bag_clf},
    {"classifier": lr_smote_clf, "name": "SGDClassifier (SMOTE)"},
    {"classifier": rf_smote_clf, "name": "RandomForestClassifier (SMOTE)"},
    {"classifier": lr_adasyn_clf, "name": "SGDClassifier (ADASYN)"},
    {"classifier": rf_adasyn_clf, "name": "RandomForestClassifier (ADASYN)"},
]


def add_classifier_result(clf, index, clf_name=None):
    clf_name = clf.__class__.__name__ if clf_name is None else clf_name
    index += [clf_name]
    cv_result = cross_validate(clf, X, Y, scoring=scoring)
    acc = cv_result["test_accuracy"].mean()
    bal_acc = cv_result["test_balanced_accuracy"].mean()
    sen = cv_result["test_sensitivity"].mean()
    pre = cv_result["test_precision"].mean()
    rec = cv_result["test_recall"].mean()
    f1 = cv_result["test_f1"].mean()
    scores["Accuracy"].append(acc)
    scores["Balanced accuracy"].append(bal_acc)
    scores["Sensitivity"].append(sen)
    scores["Precision"].append(pre)
    scores["Recall"].append(rec)
    scores["F1"].append(f1)
    logging.info(
        f"{clf_name}: acc={round(acc, 3)}, bal_acc={round(bal_acc, 3)}, sen={round(sen, 3)}, pre={round(pre, 3)}, rec={round(rec, 3)}, f1={round(f1, 3)}"
    )
    return index, scores


for clf in tqdm(classifiers):
    index, scores = add_classifier_result(
        clf["classifier"],
        index,
        clf_name=clf["name"] if "name" in clf else clf["classifier"].__class__.__name__,
    )

df_scores = pd.DataFrame(scores, index=index)
pprint(df_scores.sort_values(by=["F1"], ascending=False))
df_scores.to_csv("evaluation.csv")

logging.info("Evaluation done.")
