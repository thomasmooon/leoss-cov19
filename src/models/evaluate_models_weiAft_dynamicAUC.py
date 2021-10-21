"""
compute the dynamic AUCs for the best models
"""

# %% setup ====================================================================


from lifelines import WeibullAFTFitter
import lifelines
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from utils.general import (
    make_data_death,
    remove_special_chars,
    filter_featurecols,
    unify_ambiguous_missing)
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sksurv.metrics import cumulative_dynamic_auc
from xgbse.converters import convert_to_structured
from pycox.evaluation import EvalSurv
import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
pdir = os.path.dirname(currentdir)
ppdir = os.path.dirname(pdir)
sys.path.append(ppdir)


# %% functions ===================================================================


def convert_to_structured(T, E):
    """
    Converts data in time (T) and event (E) format to a structured numpy array.
    Provides common interface to other libraries such as sksurv and sklearn.

    Args:
        T (np.array): Array of times
        E (np.array): Array of events

    Returns:
        np.array: Structured array containing the boolean event indicator
            as first field, and time of event or time of censoring as second field
    """
    # dtypes for conversion
    default_dtypes = {"names": ("status", "time"), "formats": ("bool", "f8")}
    # concat of events and times
    concat = list(zip(E.values, T.values))
    # return structured array
    return np.array(concat, dtype=default_dtypes)


def lifelines_models_predict_survprob_testset(pipe, i, X, cv_outer):
    enc = pipe[i][0][1]
    scaler = pipe[i][1][1]
    model = pipe[i][2][1]
    _, test_idx = cv_outer[i]
    # transform
    X_t = enc.transform(X.iloc[test_idx, :])
    X_t = scaler.transform(X_t)
    X_t = pd.DataFrame(X_t, columns=enc.get_feature_names())
    X_t.reset_index(drop=True, inplace=True)
    yhat = model.predict_survival_function(X_t)
    return yhat.T


def lifelines_models_predict_medlife_testset(pipe, i, X, cv_outer):
    enc = pipe[i][0][1]
    scaler = pipe[i][1][1]
    model = pipe[i][2][1]
    _, test_idx = cv_outer[i]
    # transform
    X_t = enc.transform(X.iloc[test_idx, :])
    X_t = scaler.transform(X_t)
    X_t = pd.DataFrame(X_t, columns=enc.get_feature_names())
    X_t.reset_index(drop=True, inplace=True)
    yhat = model.predict_percentile(X_t) # median lifetimes
    return yhat


def get_ytest_outer(cv_outer, i, y):
    _, test_idx = cv_outer[i]
    return y.iloc[test_idx, :].copy()


def get_ydev_outer(cv_outer, i, y):
    dev_idx, _ = cv_outer[i]
    return y.iloc[dev_idx, :].copy()


# %% load data ====================================================================


data = pd.read_pickle("./data/processed/leoss_decoded.pkl", compression="zip")
data_items = pd.read_csv("./data/interim/data_items.txt", sep="\t")

# %% decide which data to be used further
filter_age = ["< 1 years", "4 - 8 years", "1 - 3 years",
              "15 - 17 years", "15-25 years", "> 85 years", "9 - 14 years"]
data = data[~data.BL_Age.isin(filter_age)]
X = filter_featurecols(data, data_items)
# unify unknown / missingness
X = unify_ambiguous_missing(X)
# replace na's with missing, otherwise onehotencoder fails
X.fillna("missing", inplace=True)
# preprocess endpoint specific
X, y = make_data_death(data, X)
# convert y to numpy structured array for suvival analysis
y_sa = convert_to_structured(y["time"], y["status"])

# %% cv schema
cv_outer = []
skf_outer = StratifiedKFold(n_splits=5)
for dev_idx, test_idx in skf_outer.split(X, y["status"]):
    cv_outer.append((dev_idx, test_idx))


# weibull aft ==================================================================


wei_files = [
    "/home/tlinden/leoss/mlruns/0/4457b9fcc5e94f27ad9ec1f0cf5de052/artifacts/leoss_weiAft_death_ofold0.gzip.pkl",
    "/home/tlinden/leoss/mlruns/0/56d54e39859e40ac9cf04ee2da229409/artifacts/leoss_weiAft_death_ofold1.gzip.pkl",
    "/home/tlinden/leoss/mlruns/0/d1f5444ebf1f4d0c844923b26eed3063/artifacts/leoss_weiAft_death_ofold2.gzip.pkl",
    "/home/tlinden/leoss/mlruns/0/ca051499ee6a4ff099b6d337d449ad90/artifacts/leoss_weiAft_death_ofold3.gzip.pkl",
    "/home/tlinden/leoss/mlruns/0/00be1167700d43c68ec26d0651224d2e/artifacts/leoss_weiAft_death_ofold4.gzip.pkl",
    ]
wei_models = [joblib.load(f) for f in wei_files]
wei_yhat_survprob = [lifelines_models_predict_survprob_testset(
    wei_models, i, X, cv_outer) for i in range(5)]
wei_yhat_medlife = [lifelines_models_predict_medlife_testset(
    wei_models, i, X, cv_outer) for i in range(5)]

# transform y: get dev/test slices and convert to structured array
ytest_outer = [get_ytest_outer(cv_outer, i, y) for i in range(5)]
ytest_outer_sa = [convert_to_structured(_["time"], _["status"]) for _ in ytest_outer]
ydev_outer = [get_ydev_outer(cv_outer, i, y) for i in range(5)]
ydev_outer_sa = [convert_to_structured(_["time"], _["status"]) for _ in ydev_outer]



