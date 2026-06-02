"""
This file implements several utilities for handling data I/O and implementing
regression analyses.
"""

import warnings
import os
import pickle as pkl
import copy

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.feature_selection import f_regression, SelectKBest, SelectFromModel
from sklearn.linear_model import LassoCV, Ridge, RidgeCV, ElasticNetCV
from scipy.stats import pearsonr
from sklearn.model_selection import GridSearchCV, ShuffleSplit



def list_of_ints(arg):
    return [int(a) for a in arg.split(",")]


def set_up_datasets(subject, dataset, n):
    # Load in and subset dataset
    sentences = pd.read_csv(
        os.path.join("..", "data", "processed_csvs_anon", subject, "stimset.csv")
    )["sentence"]
    betas = pd.read_csv(
        os.path.join("..", "data", "processed_csvs_anon", subject, dataset + "_betas.csv")
    )
    metadata = pd.read_csv(
        os.path.join("..", "data", "processed_csvs_anon", subject, dataset + "_meta.csv")
    )

    neuroids = list(betas.columns)[1:]
    neuroids = neuroids[:n]

    return betas, sentences, neuroids, metadata


def get_activations(model, layer, embedding_method, use_sae, sae_type=None, scale_by_decoder=False, use_topic_model=False):
    # Load in precomputed features

    if use_topic_model:
        path = f"../features/topic_models/topic_model.csv"
        activations = pd.read_csv(path).drop(columns=["item_id"]).to_numpy()
    else:
        if use_sae:
            assert sae_type is not None
            featurizer_str = f"sae/{sae_type}"
            if scale_by_decoder:
                featurizer_str += "_scaled"
        else:
            featurizer_str = "hidden_states"

        path = f"../features/{model}/{featurizer_str}/{embedding_method}/{layer}.pkl"
        activations = pkl.load(open(path, "rb"))
    return activations

def get_logprobs(model):
    # Load in precomputed logprobs
    path = f"../features/{model}/hidden_states/mean/logprobs.pkl"
    logprobs = pkl.load(open(path, "rb"))
    return logprobs

# Used for scoring
def pearson_r(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]


### Regression utils ###
def select_alpha(activations, betas, positive=False):
    # Use LOO CV or a single train-test split to select best alpha based on MSE
    ALPHAS = [10**i for i in range(-2, 6)]

    if positive:
        cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
        gs = GridSearchCV(
            Ridge(positive=positive),
            param_grid={"alpha": ALPHAS},
            cv=cv,
            scoring="neg_mean_squared_error",
        )
        gs.fit(activations, betas)
        best_alpha = gs.best_params_["alpha"]
    else:
        model = RidgeCV(alphas=ALPHAS, cv=None, fit_intercept=True)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
            )
            model.fit(activations, betas)
        best_alpha = model.alpha_
    return best_alpha


def compute_support_features(activations, betas, omit_features=[], model="lasso", k_best=8000):
    """Perform feature selection using L1-penalized regression
    on all voxels within a dataset
    """

    # First, omit any specified features by zeroing them out
    if omit_features:
        acts = copy.deepcopy(activations)
        for feature_set in omit_features:
            acts[:, feature_set] = 0
    else:
        acts = activations

    # Next, prescreen activations using univariate F-test,
    # reducing the amount of work Lasso has to do
    if acts.shape[1] > k_best:
        selector = SelectKBest(f_regression, k=k_best)
        acts_screened = selector.fit_transform(acts, betas)
        screened_indices_mask = selector.get_support() # Boolean mask of acts_subset
    else:
        acts_screened = acts
        screened_indices_mask = np.ones(acts.shape[1], dtype=bool)

    # Fit per-layer activations to responses,
    # searching to find the best alpha through
    # 5-fold CV
    cv_params = {
        "alphas": np.logspace(-2, 0, 10), # To speed things up and avoid scenarios where no features are selected
        "cv": 5,
        "tol": 1e-3,
        "selection": 'random',  # Faster for wide/sparse data
        "max_iter": 2000,
        "random_state": 19,
        "fit_intercept": True, # default, but make explicit
    }

    if model == "elasticnet":
        reg = ElasticNetCV(**cv_params)
    elif model == "lasso":
        reg = LassoCV(**cv_params)
    elif model == "lassopos":
        reg = LassoCV(**cv_params, positive=True)
    else:
        raise ValueError("Unknown feature selection model")

    reg = reg.fit(acts_screened, betas)

    # Get nonzero features with nonzero lasso regression coefficients
    lasso_support = SelectFromModel(reg, prefit=True).get_support()
    if np.sum(lasso_support) == 0:
        return None
    
    # Undo pre-screening to get feature set in original space
    support_features = np.zeros(len(acts[0]), dtype=bool)
    support_features[np.where(screened_indices_mask)[0][lasso_support]] = True

    return support_features
