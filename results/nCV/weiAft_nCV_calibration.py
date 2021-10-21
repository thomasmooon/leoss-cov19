"""
Weibull AFT nested cross-validation outer folds calibration plots
"""

# %% setup
from lifelines import WeibullAFTFitter
from lifelines.calibration import survival_probability_calibration
import lifelines
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sksurv.metrics import cumulative_dynamic_auc
from pycox.evaluation import EvalSurv
#
import sys
sys.path.append("../../")
import utils.general


# %% functions =================================================================

def get_Xy_for_lifelines_models_API(pipe, X, y, cv_outer, i):
    # unstack model components
    enc = pipe[0][1]
    scaler = pipe[1][1]
    # model = pipe[2][1]
    # get test fold index
    _, test_idx = cv_outer[i]
    # transform
    X_test = X.iloc[test_idx, :].copy()
    y_test = y.iloc[test_idx, :].copy()
    Xt = pd.DataFrame(scaler.transform(enc.transform(X_test)))
    Xt.columns = enc.get_feature_names()
    Xy = pd.concat([ Xt.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
    return Xy


# %% load and filter data ======================================================
# load data
data = pd.read_pickle("../../data/processed/leoss_decoded.pkl", compression="zip")
data_items = pd.read_csv("../../data/interim/data_items.txt", sep="\t")
# filter data
filter_age = ["< 1 years", "4 - 8 years","1 - 3 years", "15 - 17 years", "15-25 years", "> 85 years", "9 - 14 years"]
data = data[~data.BL_Age.isin(filter_age)]
X = utils.general.filter_featurecols(data, data_items)
# unify unknown / missingness
to_replace = ["Unknown","unknown", "None", "not done/unknown", "Not done/unknown", "Not determined/ Unknown"]
for rpl in to_replace:
    X.replace(rpl,"missing",inplace=True)

# replace na's with missing, otherwise onehotencoder fails
X.fillna("missing", inplace=True)
# preprocess endpoint specific
X, y = utils.general.make_data_death(data, X)


# %% cv schema
cv_outer = []
skf_outer = StratifiedKFold(n_splits=5)
for dev_idx, test_idx in skf_outer.split(X, y["status"]):
    cv_outer.append((dev_idx, test_idx))


# %% load models
# weibull aft 
files = [
    "/home/tlinden/leoss/mlruns/0/4457b9fcc5e94f27ad9ec1f0cf5de052/artifacts/leoss_weiAft_death_ofold0.gzip.pkl",
    "/home/tlinden/leoss/mlruns/0/56d54e39859e40ac9cf04ee2da229409/artifacts/leoss_weiAft_death_ofold1.gzip.pkl",
    "/home/tlinden/leoss/mlruns/0/d1f5444ebf1f4d0c844923b26eed3063/artifacts/leoss_weiAft_death_ofold2.gzip.pkl",
    "/home/tlinden/leoss/mlruns/0/ca051499ee6a4ff099b6d337d449ad90/artifacts/leoss_weiAft_death_ofold3.gzip.pkl",
    "/home/tlinden/leoss/mlruns/0/00be1167700d43c68ec26d0651224d2e/artifacts/leoss_weiAft_death_ofold4.gzip.pkl",
    ]
pipelines = [joblib.load(f) for f in files]


# %% estimate and plot test set calibrations

# plot calibrations, one panel plot per outer fold
for i in range(5):
    pipe = pipelines[i]
    aft_component = pipe[2][1]
    Xy_i = get_Xy_for_lifelines_models_API(pipe, X, y, cv_outer, i)
    # create subplots
    fig, axs= plt.subplots(nrows=2,ncols=3,sharex="col", figsize = (15,10)) # width, height
    survival_probability_calibration(aft_component, Xy_i, t0=7, ax=axs[0,0])
    survival_probability_calibration(aft_component, Xy_i, t0=14, ax=axs[0,1])
    survival_probability_calibration(aft_component, Xy_i, t0=21, ax=axs[0,2])
    survival_probability_calibration(aft_component, Xy_i, t0=28, ax=axs[1,0])
    survival_probability_calibration(aft_component, Xy_i, t0=35, ax=axs[1,1])
    survival_probability_calibration(aft_component, Xy_i, t0=80, ax=axs[1,2])
    plt.suptitle(f"test set performance, fold {i}")
    plt.savefig(f"weiAft_nCV_survival_probability_calibration_fold{i}.pdf", bbox_inches="tight")
    plt.close()



