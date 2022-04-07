# %% [markdown]
# # Principal Component Analysis for BERT features
#
# Based on [Model selection with Probabilistic PCA and Factor Analysis (FA)](https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_fa_model_selection.html#sphx-glr-auto-examples-decomposition-plot-pca-vs-fa-model-selection-py) by scikit-learn.

# %%
# Authors: Alexandre Gramfort
#          Denis A. Engemann
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import json
from tqdm import tqdm

from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.covariance import ShrunkCovariance, LedoitWolf
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# #############################################################################
# Load the data

data_path = "../Screen2Vec/simplifiedscreen2vec/data/links_pca.json"
with open(data_path) as data_json:
    links = json.load(data_json)["links"]

X = np.array(
    [
        np.array(sample["link"]["source"]["element"]["embedding"])
        - np.array(sample["link"]["target"]["embedding"]["text"])
        for sample in links
    ]
)
np.shape(X)

# %%

# #############################################################################
# Fit the models

n_components = np.arange(50, 101, 1)  # options for n_components


def compute_scores(X):
    pca = PCA(svd_solver="full")
    fa = FactorAnalysis()

    pca_scores, fa_scores = [], []
    for n in tqdm(n_components):
        pca.n_components = n
        fa.n_components = n
        pca_scores.append(np.mean(cross_val_score(pca, X)))
        fa_scores.append(np.mean(cross_val_score(fa, X)))

        print("current n = %d" % n)
        n_components_pca = n_components[np.argmax(pca_scores)]
        n_components_fa = n_components[np.argmax(fa_scores)]
        print("best n_components by PCA CV = %d" % n_components_pca)
        print("best n_components by FactorAnalysis CV = %d" % n_components_fa)

    return pca_scores, fa_scores


def shrunk_cov_score(X):
    shrinkages = np.logspace(-2, 0, 30)
    cv = GridSearchCV(ShrunkCovariance(), {"shrinkage": shrinkages})
    return np.mean(cross_val_score(cv.fit(X).best_estimator_, X))


def lw_score(X):
    return np.mean(cross_val_score(LedoitWolf(), X))


pca_scores, fa_scores = compute_scores(X)
n_components_pca = n_components[np.argmax(pca_scores)]
n_components_fa = n_components[np.argmax(fa_scores)]

pca = PCA(svd_solver="full", n_components="mle")
pca.fit(X)
n_components_pca_mle = pca.n_components_

print("best n_components by PCA CV = %d" % n_components_pca)
print("best n_components by FactorAnalysis CV = %d" % n_components_fa)
print("best n_components by PCA MLE = %d" % n_components_pca_mle)

# %%
plt.figure()
plt.plot(n_components, pca_scores, "b", label="PCA scores")
plt.plot(n_components, fa_scores, "r", label="FA scores")
plt.axvline(
    n_components_pca,
    color="b",
    label="PCA CV: %d" % n_components_pca,
    linestyle="--",
)
plt.axvline(
    n_components_fa,
    color="r",
    label="FactorAnalysis CV: %d" % n_components_fa,
    linestyle="--",
)
plt.axvline(
    n_components_pca_mle,
    color="k",
    label="PCA MLE: %d" % n_components_pca_mle,
    linestyle="--",
)

# compare with other covariance estimators
plt.axhline(
    shrunk_cov_score(X),
    color="violet",
    label="Shrunk Covariance MLE",
    linestyle="-.",
)
plt.axhline(
    lw_score(X),
    color="orange",
    label="LedoitWolf MLE" % n_components_pca_mle,
    linestyle="-.",
)

plt.xlabel("nb of components")
plt.ylabel("CV scores")
plt.legend(loc="lower right")
plt.title("Text Similarity")

plt.savefig("pca.png")
plt.show()

# %%

# #############################################################################
# Save the best PCA model

from joblib import dump

pca = PCA(svd_solver="full", n_components=n_components_pca)
pca.fit(X)
dump(pca, "pca.joblib")
