"""
evaluate metrics:
- Harrells C-index
- Harrells time-dependent C-index
- D-Calibration
- IBS

# srun -p gpu -n 1  --gres gpu:v100:1 --cpus-per-task=5 --mem-per-cpu=32gb --time=16:00:00 --pty /bin/bash
# echo "set enable-bracketed-paste off" >> ~/.inputrc
"""

# %% setup ====================================================================
from xgbse import XGBSEStackedWeibull
from lifelines import WeibullAFTFitter
from lifelines import CoxPHFitter
import lifelines
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from utils.general import (
    make_data_death,
    make_data_critical,
    remove_special_chars,
    filter_featurecols,
    unify_ambiguous_missing)
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sksurv.metrics import concordance_index_ipcw as cuno
from sksurv.metrics import brier_score, cumulative_dynamic_auc
from xgbse.converters import convert_to_structured
from xgbse.metrics import concordance_index, approx_brier_score, dist_calibration_score
from pycox.evaluation import EvalSurv
import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
pdir = os.path.dirname(currentdir)
ppdir = os.path.dirname(pdir)
sys.path.append(ppdir)


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


# %% functions =================================================================


def lifelines_models_predict_testset(pipe, i, X, cv_outer):
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


def xgbse_predict_testset(pipe, i, X, cv_outer):
    enc = pipe[i][0][1]
    model = pipe[i][1][1]
    _, test_idx = cv_outer[i]
    # transform
    X_t = enc.transform(X.iloc[test_idx, :])
    X_t = pd.DataFrame(
        X_t, columns=remove_special_chars(enc.get_feature_names()))
    yhat = model.predict(X_t)
    return yhat


def rsf_predict_testset(optuna_search, i, X, cv_outer):
    r"""
    tested in C:\Users\tlinden\Documents\Git\leoss\notebooks\test_rsf_prediction.ipynb
    """
    _, test_idx = cv_outer[i]
    enc = optuna_search[i].best_estimator_[0]
    rsf = optuna_search[i].best_estimator_[1]
    yhat = rsf.predict_survival_function(enc.transform(X.iloc[test_idx, :]))
    yhat_sprob = [fn(fn.x) for fn in yhat]
    yhat_sprob = pd.DataFrame(yhat_sprob)
    return yhat_sprob


def deepsurv_predict_testset_hazard(transformations_and_model, i, X, cv_outer):
    # TODO estimate baseline hazard from dev_set y
    _, test_idx = cv_outer[i]
    coltransformer = transformations_and_model[i][0]
    model = transformations_and_model[i][1]
    yhat = model.predict(coltransformer.transform(
        X.iloc[test_idx, :]).astype("float32"))
    return yhat


def deepsurv_predict_testset_survprob(transformations_and_model, i, X, cv_outer):
    # slice data
    _, test_idx = cv_outer[i]
    X_test = X.iloc[test_idx, :]
    # do deepsurv predictions
    coltransformer = transformations_and_model[i][0]
    model = transformations_and_model[i][1]
    yhat_test = model.predict_surv_df(
        coltransformer.transform(X_test).astype("float32"))
    return yhat_test.T


def get_ytest_outer(cv_outer, i, y):
    _, test_idx = cv_outer[i]
    return y.iloc[test_idx, :].copy()


def get_ydev_outer(cv_outer, i, y):
    dev_idx, _ = cv_outer[i]
    return y.iloc[dev_idx, :].copy()


# rsf ==========================================================================


rsf_files = [
    "/home/tlinden/leoss/mlruns/0/c79f016e8f404383beaec4637fc21c95/artifacts/leoss_rsf_death_ofold0_optuna_search.pkl",
    "/home/tlinden/leoss/mlruns/0/3ba3b87a7efa4e0b960b9c6ed34bc547/artifacts/leoss_rsf_death_ofold1_optuna_search.pkl",
    "/home/tlinden/leoss/mlruns/0/328bcf7a92ca49ad98e3c78d87692cfd/artifacts/leoss_rsf_death_ofold2_optuna_search.pkl",
    "/home/tlinden/leoss/mlruns/0/e5b5cc45dbbc4cc4a9ffeabd68d312fe/artifacts/leoss_rsf_death_ofold3_optuna_search.pkl",
    "/home/tlinden/leoss/mlruns/0/6f7006578cc14fe5b241fb387fa33613/artifacts/leoss_rsf_death_ofold4_optuna_search.pkl",
]
rsf_models = [joblib.load(f) for f in rsf_files]
rsf_yhat = [rsf_predict_testset(rsf_models, i, X, cv_outer) for i in range(5)]


# deepsurv =====================================================================


deepsurv_files = [
    "/home/tlinden/leoss/mlruns/0/37ebdc81cde047828ee70832f52bb205/artifacts/leoss_deepsurv_death_ofold0_bestmodel.pkl",
    "/home/tlinden/leoss/mlruns/0/0dde06bb0380446abe864434791fed95/artifacts/leoss_deepsurv_death_ofold1_bestmodel.pkl",
    "/home/tlinden/leoss/mlruns/0/0f1de896284c463783905949def7511d/artifacts/leoss_deepsurv_death_ofold2_bestmodel.pkl",
    "/home/tlinden/leoss/mlruns/0/c2ba04736f6449d493460201cafeef8e/artifacts/leoss_deepsurv_death_ofold3_bestmodel.pkl",
    "/home/tlinden/leoss/mlruns/0/b105b8710b7b41efa41eae724fd45337/artifacts/leoss_deepsurv_death_ofold4_bestmodel.pkl"
]
deepsurv_models = [joblib.load(f) for f in deepsurv_files]
deepsurv_yhat_hazards = [deepsurv_predict_testset_hazard(
    deepsurv_models, i, X, cv_outer) for i in range(5)]
deepsurv_yhat_surv = [deepsurv_predict_testset_survprob(
    deepsurv_models, i, X, cv_outer) for i in range(5)]


# cox ==========================================================================


cox_files = [
    "/home/tlinden/leoss/mlruns/0/1d934afa47c84f488a80742ecc5a241d/artifacts/leoss_cox_death_ofold0.gzip.pkl",
    "/home/tlinden/leoss/mlruns/0/41e88c1ea07042f99bdff4d62dfeb0d1/artifacts/leoss_cox_death_ofold1.gzip.pkl",
    "/home/tlinden/leoss/mlruns/0/a282301cdcc14b11ba5113a422bcde4e/artifacts/leoss_cox_death_ofold2.gzip.pkl",
    "/home/tlinden/leoss/mlruns/0/cc52f3bd6ab74bba82fba6a63ff883cc/artifacts/leoss_cox_death_ofold3.gzip.pkl",
    "/home/tlinden/leoss/mlruns/0/a10a8f8c9c9947ceb556094f9cef7064/artifacts/leoss_cox_death_ofold4.gzip.pkl",
]
cox_models = [joblib.load(f) for f in cox_files]
cox_yhat = [lifelines_models_predict_testset(
    cox_models, i, X, cv_outer) for i in range(5)]


# weibull aft ==================================================================


wei_files = [
    "/home/tlinden/leoss/mlruns/0/4457b9fcc5e94f27ad9ec1f0cf5de052/artifacts/leoss_weiAft_death_ofold0.gzip.pkl",
    "/home/tlinden/leoss/mlruns/0/56d54e39859e40ac9cf04ee2da229409/artifacts/leoss_weiAft_death_ofold1.gzip.pkl",
    "/home/tlinden/leoss/mlruns/0/d1f5444ebf1f4d0c844923b26eed3063/artifacts/leoss_weiAft_death_ofold2.gzip.pkl",
    "/home/tlinden/leoss/mlruns/0/ca051499ee6a4ff099b6d337d449ad90/artifacts/leoss_weiAft_death_ofold3.gzip.pkl",
    "/home/tlinden/leoss/mlruns/0/00be1167700d43c68ec26d0651224d2e/artifacts/leoss_weiAft_death_ofold4.gzip.pkl",
]
wei_models = [joblib.load(f) for f in wei_files]
wei_yhat = [lifelines_models_predict_testset(
    wei_models, i, X, cv_outer) for i in range(5)]


# xgbse-AFT ===================================================================


xgbse_files = [
    "/home/tlinden/leoss/mlruns/0/0bf8d0354cef41d88c45b3ee7b80b2ee/artifacts/leoss_xgbse_sWeiAft_death_ofold0.gzip.pkl",
    "/home/tlinden/leoss/mlruns/0/215e943df848452ea76d87d4aadce604/artifacts/leoss_xgbse_sWeiAft_death_ofold1.gzip.pkl",
    "/home/tlinden/leoss/mlruns/0/61cd96beb4fe4054a688f285f16a67bf/artifacts/leoss_xgbse_sWeiAft_death_ofold2.gzip.pkl",
    "/home/tlinden/leoss/mlruns/0/f3ccb3dec4e74859b1bc70262ca35bba/artifacts/leoss_xgbse_sWeiAft_death_ofold3.gzip.pkl",
    "/home/tlinden/leoss/mlruns/0/bd8b21d7ea72469f81603b7202f76c40/artifacts/leoss_xgbse_sWeiAft_death_ofold4.gzip.pkl",
]
xgbse_models = [joblib.load(f) for f in xgbse_files]
xgbse_yhat = [xgbse_predict_testset(
    xgbse_models, i, X, cv_outer) for i in range(5)]

# %% evaluate =================================================================

# transform y: get dev/test slices and convert to structured array
ytest_outer = [get_ytest_outer(cv_outer, i, y) for i in range(5)]
ytest_outer_sa = [convert_to_structured(_["time"], _["status"]) for _ in ytest_outer]
ydev_outer = [get_ydev_outer(cv_outer, i, y) for i in range(5)]
ydev_outer_sa = [convert_to_structured(_["time"], _["status"]) for _ in ydev_outer]


# harrells c
rsf_charrell = [concordance_index(
    ytest_outer_sa[i], rsf_yhat[i]) for i in range(5)]
deepsurv_charrell = [concordance_index(
    ytest_outer_sa[i], 1-deepsurv_yhat_hazards[i]) for i in range(5)]
cox_charrell = [concordance_index(
    ytest_outer_sa[i], cox_yhat[i]) for i in range(5)]
wei_charrell = [concordance_index(
    ytest_outer_sa[i], wei_yhat[i]) for i in range(5)]
xgbse_charrell = [concordance_index(
    ytest_outer_sa[i], xgbse_yhat[i]) for i in range(5)]

# integrated brier score
rsf_ibs = [approx_brier_score(ytest_outer_sa[i], rsf_yhat[i])
           for i in range(5)]
deepsurv_ibs = [approx_brier_score(
    ytest_outer_sa[i], deepsurv_yhat_surv[i]) for i in range(5)]
cox_ibs = [approx_brier_score(ytest_outer_sa[i], cox_yhat[i])
           for i in range(5)]
wei_ibs = [approx_brier_score(ytest_outer_sa[i], wei_yhat[i])
           for i in range(5)]
xgbse_ibs = [approx_brier_score(
    ytest_outer_sa[i], xgbse_yhat[i]) for i in range(5)]

# calibration
rsf_dcal_p = [dist_calibration_score(
    ytest_outer_sa[i], rsf_yhat[i], n_bins = 100) for i in range(5)]
rsf_dcal_max = [dist_calibration_score(
    ytest_outer_sa[i], rsf_yhat[i], n_bins = 100, returns="max_deviation") for i in range(5)]
deepsurv_dcal_p = [dist_calibration_score(
    ytest_outer_sa[i], deepsurv_yhat_surv[i], n_bins = 100) for i in range(5)]
deepsurv_dcal_max = [dist_calibration_score(
    ytest_outer_sa[i], deepsurv_yhat_surv[i], n_bins = 100, returns="max_deviation") for i in range(5)]
cox_dcal_p = [dist_calibration_score(
    ytest_outer_sa[i], cox_yhat[i], n_bins = 100) for i in range(5)]
cox_dcal_max = [dist_calibration_score(
    ytest_outer_sa[i], cox_yhat[i], n_bins = 100, returns="max_deviation") for i in range(5)]
xgbse_dcal_p = [dist_calibration_score(
    ytest_outer_sa[i], xgbse_yhat[i], n_bins = 100) for i in range(5)]
xgbse_dcal_max = [dist_calibration_score(
    ytest_outer_sa[i], xgbse_yhat[i], n_bins = 100, returns="max_deviation") for i in range(5)]
wei_dcal_p = [dist_calibration_score(
    ytest_outer_sa[i], wei_yhat[i], n_bins = 100) for i in range(5)]
wei_dcal_max = [dist_calibration_score(
    ytest_outer_sa[i], wei_yhat[i], n_bins = 100, returns="max_deviation") for i in range(5)]

# concordance td
def EvalSurv_(data, i): 
    ctd = EvalSurv(
        data[i].T, 
        ytest_outer[i]["time"].to_numpy(),
        ytest_outer[i]["status"].to_numpy(),
        censor_surv='km').concordance_td()
    return ctd

deepsurv_ctd = [EvalSurv_(deepsurv_yhat_surv, i) for i in range(5)]
cox_ctd = [EvalSurv_(cox_yhat, i) for i in range(5)]
wei_ctd = [EvalSurv_(wei_yhat, i) for i in range(5)]
xgbse_ctd = [EvalSurv_(xgbse_yhat, i) for i in range(5)]
deepsurv_ctd = [EvalSurv_(xgbse_yhat, i) for i in range(5)]
rsf_ctd = [EvalSurv_(xgbse_yhat, i) for i in range(5)]

# organize results
metrics = {
    "cox": {
        "charrell": cox_charrell,
        "ctd": cox_ctd,
        "ibs": cox_ibs,
        "dcal_p": cox_dcal_p,
        "dcal_max": cox_dcal_max,
        "model": "cox"
    },
    "wei": {
        "charrell": wei_charrell,
        "ctd": wei_ctd,
        "ibs": wei_ibs,
        "dcal_p": wei_dcal_p,
        "dcal_max": wei_dcal_max,
        "model": "wei"
    },
    "xgbse": {
        "charrell": xgbse_charrell,
        "ctd": xgbse_ctd,
        "ibs": xgbse_ibs,
        "dcal_p": xgbse_dcal_p,
        "dcal_max": xgbse_dcal_max,
        "model": "xgbse"
    },
    "rsf": {
        "charrell": rsf_charrell,
        "ctd": rsf_ctd,
        "ibs": rsf_ibs,
        "dcal_p": rsf_dcal_p,
        "dcal_max": rsf_dcal_max,
        "model": "rsf"
    },
    "deepsurv": {
        "charrell": deepsurv_charrell,
        "ctd": deepsurv_ctd,
        "ibs": deepsurv_ibs,
        "dcal_p": deepsurv_dcal_p,
        "dcal_max": deepsurv_dcal_max,
        "model": "deepsurv"
    },
}

metrics_nCV = [pd.DataFrame(v) for k, v in metrics.items()]
metrics_nCV = pd.concat(metrics_nCV, axis=0)
metrics_nCV.to_csv("results/nCV/metrics_nCV.csv", sep="\t")
