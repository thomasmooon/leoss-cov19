"""
evaluate metrics: 
- cUno
- cUno(t)

CAUTION ---> run on GPU, otherwise inconvenience when re-loading deepSurv models
# srun -p gpu -n 1  --gres gpu:v100:1 --time=8:00:00 --pty /bin/bash
# echo "set enable-bracketed-paste off" >> ~/.inputrc
"""

# %% setup
from os import error
from xgbse import XGBSEStackedWeibull
from lifelines import WeibullAFTFitter
from lifelines import CoxPHFitter
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
from sksurv.metrics import concordance_index_ipcw as cuno
from sksurv.metrics import cumulative_dynamic_auc
from xgbse.converters import convert_to_structured


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
    event_times_ = rsf.event_times_.tolist()
    yhat = rsf.predict_survival_function(enc.transform(X.iloc[test_idx, :]))
    yhat_sprob = [fn(fn.x) for fn in yhat]
    yhat_sprob = pd.DataFrame(yhat_sprob, columns=event_times_)
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


def cuno_try(ydev_outer_sa, ytest_outer_sa, estimate, i):
    """Computing cuno can fail dependent on tau, hence iterate through times
    (in ascending order) to try out different times
    """
    times = np.unique(ydev_outer_sa[i]["c2"])
    times = np.flip(times)
    for t_ in times:
        try:
            cuno_ = cuno(ydev_outer_sa[i], ytest_outer_sa[i], estimate, tau=t_)[0]
            print(f"time {t_} SUCCESS with cuno = {cuno_}")
            break
        except:
            print(f"time {t_} FAILED")
    return cuno_


# %% adapt original cumulative_dynamic_auc method to support truncation times

from sklearn.utils import check_array, check_consistent_length
from sksurv.util import check_y_survival
from sksurv.nonparametric import CensoringDistributionEstimator, SurvivalFunctionEstimator


def _check_times(test_time, times):
    times = check_array(np.atleast_1d(times), ensure_2d=False, dtype=test_time.dtype)
    times = np.unique(times)
    if times.max() >= test_time.max() or times.min() < test_time.min():
        raise ValueError(
            'all times must be within follow-up time of test data: [{}; {}['.format(
                test_time.min(), test_time.max()))
    return times


def _check_estimate_2d(estimate, test_time, time_points):
    estimate = check_array(estimate, ensure_2d=False, allow_nd=False)
    time_points = _check_times(test_time, time_points)
    check_consistent_length(test_time, estimate)
    if estimate.ndim == 2 and estimate.shape[1] != time_points.shape[0]:
        raise ValueError("expected estimate with {} columns, but got {}".format(
            time_points.shape[0], estimate.shape[1]))
    return estimate, time_points


def my_cumulative_dynamic_auc(survival_train, survival_test, estimate, times, tau=None, tied_tol=1e-8):
    """modified  https://github.com/sebp/scikit-survival/blob/master/sksurv/metrics.py#L330-L501
    with contents from https://github.com/sebp/scikit-survival/blob/master/sksurv/metrics.py#L224-L327
    commit a67ebf448f4814f1e9e0ffefba89e8f7f73983cb
    """
    test_event, test_time = check_y_survival(survival_test)    
    # from https://github.com/sebp/scikit-survival/blob/master/sksurv/metrics.py#L309-L311
    # XXX -->
    if tau is not None: 
        mask = test_time < tau 
        survival_test = survival_test[mask]
    # XXX <--
    estimate, times = _check_estimate_2d(estimate, test_time, times)
    #
    n_samples = estimate.shape[0]
    n_times = times.shape[0]
    if estimate.ndim == 1:
        estimate = np.broadcast_to(estimate[:, np.newaxis], (n_samples, n_times))
    #
    # fit and transform IPCW
    cens = CensoringDistributionEstimator()
    cens.fit(survival_train)
    # ipcw = cens.predict_ipcw(survival_test) # XXX
    ipcw_test = cens.predict_ipcw(survival_test) # XXX
    #
    # from https://github.com/sebp/scikit-survival/blob/master/sksurv/metrics.py#L318-L323
    # XXX -->
    if tau is None:
        ipcw = ipcw_test
    else:
        ipcw = np.empty(estimate.shape[0], dtype=ipcw_test.dtype)
        ipcw[mask] = ipcw_test
        ipcw[~mask] = 0
    # XXX <--
    # expand arrays to (n_samples, n_times) shape
    test_time = np.broadcast_to(test_time[:, np.newaxis], (n_samples, n_times))
    test_event = np.broadcast_to(test_event[:, np.newaxis], (n_samples, n_times))
    times_2d = np.broadcast_to(times, (n_samples, n_times))
    ipcw = np.broadcast_to(ipcw[:, np.newaxis], (n_samples, n_times))
    # sort each time point (columns) by risk score (descending)
    o = np.argsort(-estimate, axis=0)
    test_time = np.take_along_axis(test_time, o, axis=0)
    test_event = np.take_along_axis(test_event, o, axis=0)
    estimate = np.take_along_axis(estimate, o, axis=0)
    ipcw = np.take_along_axis(ipcw, o, axis=0)
    is_case = (test_time <= times_2d) & test_event
    is_control = test_time > times_2d
    n_controls = is_control.sum(axis=0)
    # prepend row of infinity values
    estimate_diff = np.concatenate((np.broadcast_to(np.infty, (1, n_times)), estimate))
    is_tied = np.absolute(np.diff(estimate_diff, axis=0)) <= tied_tol
    cumsum_tp = np.cumsum(is_case * ipcw, axis=0)
    cumsum_fp = np.cumsum(is_control, axis=0)
    true_pos = cumsum_tp / cumsum_tp[-1]
    false_pos = cumsum_fp / n_controls
    #
    scores = np.empty(n_times, dtype=float)
    it = np.nditer((true_pos, false_pos, is_tied), order="F", flags=["external_loop"])
    with it:
        for i, (tp, fp, mask) in enumerate(it):
            idx = np.flatnonzero(mask) - 1
            # only keep the last estimate for tied risk scores
            tp_no_ties = np.delete(tp, idx)
            fp_no_ties = np.delete(fp, idx)
            # Add an extra threshold position
            # to make sure that the curve starts at (0, 0)
            tp_no_ties = np.r_[0, tp_no_ties]
            fp_no_ties = np.r_[0, fp_no_ties]
            scores[i] = np.trapz(tp_no_ties, fp_no_ties)
    #
    if n_times == 1:
        mean_auc = scores[0]
    else:
        surv = SurvivalFunctionEstimator()
        surv.fit(survival_test)
        s_times = surv.predict_proba(times)
        # compute integral of AUC over survival function
        d = -np.diff(np.r_[1.0, s_times])
        integral = (scores * d).sum()
        mean_auc = integral / (1.0 - s_times[-1])
    #
    return scores, mean_auc


def cumulative_dynamic_auc_try2(survival_train , survival_test, estimate, times_estimate):
    """Computing cuno can fail dependent on tau, hence iterate through times
    (in ascending order) to try out different times
    """
    # times_train = np.unique(survival_train["c2"]) 
    # times_test = np.unique(survival_test["c2"])
    # times_joint = np.array(list(set(times_train) & set(times_test)))
    # times = np.flip(times_joint)
    times_estimate = np.array(times_estimate)
    times = list(sorted(set(survival_test["c2"])))        
    times = np.array(times)    
    for tau in np.flip(times):
        try:
            cda = my_cumulative_dynamic_auc(
                survival_train, 
                survival_test, 
                estimate.iloc[:,times_estimate<tau],
                times_estimate[times_estimate<tau],
                tau,                
                )
            print(f"tau {tau} SUCCESS")
            break
        except Exception as e:
            print(e)
            print(f"tau {tau} FAILED")    
    return {"times": times_estimate[times_estimate<tau], "auc": cda[0], "iauc" : cda[1]}


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


eepsurv_files = [
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


# xgbse-AFT ====================================================================


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


#%% transform y ===============================================================


# get dev/test slices and convert to structured array
ytest_outer = [get_ytest_outer(cv_outer, i, y) for i in range(5)]
ytest_outer_sa = [convert_to_structured(_["time"], _["status"]) for _ in ytest_outer]
ydev_outer = [get_ydev_outer(cv_outer, i, y) for i in range(5)]
ydev_outer_sa = [convert_to_structured(_["time"], _["status"]) for _ in ydev_outer]


# %% evaluate: cUno ===========================================================

deepsurv_cuno = []
for i in range(5):
    deepsurv_cuno.append(
        cuno_try(
            ydev_outer_sa, 
            ytest_outer_sa, 
            deepsurv_yhat_hazards[i].flatten(),
            i
            ))

rsf_cuno = []
for i in range(5):
    rsf_cuno.append(
        cuno_try(
            ydev_outer_sa, 
            ytest_outer_sa, 
            1-np.mean(rsf_yhat[i],1),
            i
            ))

cox_cuno = []
for i in range(5):
    cox_cuno.append(
        cuno_try(
            ydev_outer_sa, 
            ytest_outer_sa, 
            1-np.mean(cox_yhat[i],1),
            i
            ))

wei_cuno = []
for i in range(5):
    wei_cuno.append(
        cuno_try(
            ydev_outer_sa, 
            ytest_outer_sa, 
            1-np.mean(wei_yhat[i],1),
            i
            ))

xgbse_cuno = []
for i in range(5):
    xgbse_cuno.append(
        cuno_try(
            ydev_outer_sa, 
            ytest_outer_sa, 
            1-np.mean(xgbse_yhat[i],1),
            i
            ))


# organize results 
metrics = {
    "cox": {
        "cuno": cox_cuno,
        "model": "cox"
    },
    "wei": {
        "cuno": wei_cuno,
        "model": "wei"
    },
    "xgbse": {
        "cuno": xgbse_cuno,
        "model": "xgbse"
    },
    "rsf": {
        "cuno": rsf_cuno,
        "model": "rsf"
    },
    "deepsurv": {
        "cuno": deepsurv_cuno,
        "model": "deepsurv"
    },
}

metrics_nCV_cuno = [pd.DataFrame(v) for k, v in metrics.items()]
metrics_nCV_cuno = pd.concat(metrics_nCV_cuno, axis=0)
metrics_nCV_cuno.to_csv("results/nCV/metrics_nCV_cuno.csv", sep="\t")
joblib.dump(metrics_nCV_cuno, "results/nCV/metrics_nCV_cuno.joblib.zip", compress=5)


# %% evaluate: cumulative dynamic AUC =========================================


deepsurv_dynuno = []
for i in range(5):
    deepsurv_dynuno.append(
        cumulative_dynamic_auc_try2(
            ydev_outer_sa[i], 
            ytest_outer_sa[i], 
            1-deepsurv_yhat_surv[i],
            deepsurv_yhat_surv[i].columns.values.tolist()
            ))

rsf_dynuno = []
for i in range(5):
    rsf_dynuno.append(
        cumulative_dynamic_auc_try2(
            ydev_outer_sa[i], 
            ytest_outer_sa[i], 
            1-rsf_yhat[i],
            rsf_yhat[i].columns.values.tolist()
            ))

cox_dynuno = []
for i in range(5):
    cox_dynuno.append(
        cumulative_dynamic_auc_try2(
            ydev_outer_sa[i], 
            ytest_outer_sa[i], 
            1-cox_yhat[i],
            cox_yhat[i].columns.values.tolist()
            ))

wei_dynuno = []
for i in range(5):
    wei_dynuno.append(
        cumulative_dynamic_auc_try2(
            ydev_outer_sa[i], 
            ytest_outer_sa[i], 
            1-wei_yhat[i],
            wei_yhat[i].columns.values.tolist()
            ))

xgbse_dynuno = []
for i in range(5):
    xgbse_dynuno.append(
        cumulative_dynamic_auc_try2(
            ydev_outer_sa[i], 
            ytest_outer_sa[i], 
            1-xgbse_yhat[i],
            xgbse_yhat[i].columns.values.tolist()
            ))

# add metadata
def add_meta(x : "List[Dict]", key, value):
    for i in range(len(x)):
        x[i].update({key:value})
        x[i].update({"fold":i})
    return x

deepsurv_dynuno = add_meta(deepsurv_dynuno, "model","deepsurv")
rsf_dynuno = add_meta(rsf_dynuno, "model","rsf")
cox_dynuno = add_meta(cox_dynuno, "model","cox")
wei_dynuno = add_meta(wei_dynuno, "model","wei")
xgbse_dynuno = add_meta(xgbse_dynuno, "model","xgbse")

# concat
deepsurv_dynuno = pd.concat([pd.DataFrame(_) for _ in deepsurv_dynuno])
rsf_dynuno = pd.concat([pd.DataFrame(_) for _ in rsf_dynuno])
cox_dynuno = pd.concat([pd.DataFrame(_) for _ in cox_dynuno])
wei_dynuno = pd.concat([pd.DataFrame(_) for _ in wei_dynuno])
xgbse_dynuno = pd.concat([pd.DataFrame(_) for _ in xgbse_dynuno])
dynuno = pd.concat([deepsurv_dynuno, rsf_dynuno, cox_dynuno, wei_dynuno, xgbse_dynuno])

# save results
dynuno.to_csv("results/nCV/metrics_nCV_dynuno.csv", sep="\t")
joblib.dump(dynuno, "results/nCV/metrics_nCV_dynuno.joblib.zip", compress=5)
