"""
XGB-AFT nested 5x5-fold cross-validation
"""

# %% argparse =================================================================


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--ofold",help="integer id of the outer fold allowed: [0,1,2,3,4]", type=int, nargs="*", default=[0,1,2,3,4])
parser.add_argument("--run_id", help="mlflow parent run run-id", type=str, default=None)
args = parser.parse_args()
print(args)


# %% setup ========================================================================

import mlflow
import numpy as np
import pandas as pd
import pickle, joblib
import optuna
from pprint import pprint
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from sksurv.metrics import concordance_index_censored as cindex

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
pdir = os.path.dirname(currentdir)
ppdir = os.path.dirname(pdir)
sys.path.append(ppdir)
import utils.general
from utils.general import (
    make_data_death, 
    filter_featurecols, 
    unify_ambiguous_missing
    )

# constants

# MODUS = "dev"
MODUS = "experiment"
CV_NSPLITS=5
ENDPOINT = "death"
MODEL = "xgbAft"
RANDOM_STATE=42
OPTUNA_NTRIALS=40
OPTUNA_NJOBS=2
# XGB_AFT_LOSS_DISTRIBUTION_SCALE has high impact on calibration, 
# see https://towardsdatascience.com/xgbse-improving-xgboost-for-survival-analysis-393d47f1384a
# not tuned here, because marginal effect on c-index. moreover xgb does only output
# a scalar (event time) and no risk-over-time. hence only suitable for ranking, hence
# poor calibration would NOT be a problem.
XGB_AFT_LOSS_DISTRIBUTION_SCALE = 1.2
XGB_MAX_NUM_BOOST_ROUND = 1000
XGB_EARLY_STOPPING_ROUNDS = 10
XGB_NTHREAD = 1
XGB_BOOSTERPARAMS = {
    'objective': 'survival:aft',
    'eval_metric': 'aft-nloglik',
    'aft_loss_distribution': 'normal',
    'nthread' : XGB_NTHREAD,
    'aft_loss_distribution_scale': XGB_AFT_LOSS_DISTRIBUTION_SCALE, 
    'tree_method': 'hist',
    }


# %% load data ====================================================================


data = pd.read_pickle("./data/processed/leoss_decoded.pkl", compression="zip")
data_items = pd.read_csv("./data/interim/data_items.txt", sep="\t")

#%% decide which data to be used further
filter_age = [
    "< 1 years", "4 - 8 years","1 - 3 years", "15 - 17 years", "15-25 years",
    "> 85 years", "9 - 14 years"]
data = data[~data.BL_Age.isin(filter_age)]
X = filter_featurecols(data, data_items)
# unify unknown / missingness
X = unify_ambiguous_missing(X)
# replace na's with missing, otherwise onehotencoder fails
X.fillna("missing", inplace=True)
# preprocess endpoint specific
X, y = make_data_death(data, X)


# this is not required for xgb, but reflects approach used for cox and weibull_aft
# for consistency:
# get all possible categories and drop 'first' to avoid multi-colinearity issues
# during model training
oh_ = OneHotEncoder(drop = "if_binary")
oh_.fit_transform(X)
oh_all_categories = oh_.categories_


#%% optuna objective function ==================================================


class Objective(object):      
    
    def __init__(
        self, 
        X: pd.DataFrame,
        y: pd.DataFrame,
        max_num_boost_round=1000,
        cv=5,
        early_stopping_rounds = 10, 
        nthread = -1,
        outer_fold="NA",
        boosterparams = dict(),
        oh_all_categories = oh_all_categories
        ) -> float:

        self.X = X
        self.y = y
        self.max_num_boost_round = max_num_boost_round
        self.cv = cv
        self.early_stopping_rounds = early_stopping_rounds
        self.nthread = nthread
        self.outer_fold = outer_fold
        self.boosterparams = boosterparams
        self.oh_all_categories = oh_all_categories


    def __call__(self, trial) -> float:
        X = self.X
        y = self.y        

        # (hyper)parameters
        params = {
            'learning_rate': trial.suggest_loguniform("learning_rate", 1e-1, 1.0),
            'max_depth': trial.suggest_int("max_depth", 2, 6),
            'subsample' : trial.suggest_uniform("subsample",0.5,1),
            'num_parallel_tree' : trial.suggest_int("num_parallel_tree", 2, 10),
            "rate_drop" : trial.suggest_loguniform("rate_drop", 1e-3, 1e-2) 
            }        
        params.update(self.boosterparams)

        train_scores = []
        val_scores = []        
        # one inner stratified cross validation loop
        for i, idx in enumerate(cv_inner):
            # subset data wrt cv-fold
            train_idx, val_idx = idx
            X_train = X.iloc[train_idx,:]
            X_val = X.iloc[val_idx,:]
            y_train = y.iloc[train_idx,:]
            y_val = y.iloc[val_idx,:]
            # one hot encoding because xgb does not support categorical variables
            enc = OneHotEncoder(sparse=False, drop="if_binary", categories = self.oh_all_categories)
            X_train_oh = enc.fit_transform(X_train)
            X_val_oh = enc.transform(X_val)
            X_train_oh = pd.DataFrame(X_train_oh, columns=enc.get_feature_names())
            X_val_oh = pd.DataFrame(X_val_oh, columns=enc.get_feature_names())
            # make xgb data
            dtrain = utils.general.xgb_make_survdmatrix(X_train_oh, y_train, label = "train")
            dval = utils.general.xgb_make_survdmatrix(X_val_oh, y_val, label = "val")

            if MODUS == "experiment":
                model = xgb.train(
                    params = params, 
                    dtrain = dtrain,
                    num_boost_round = self.max_num_boost_round, 
                    evals = [(dval,"val")], 
                    early_stopping_rounds = self.early_stopping_rounds,
                    verbose_eval=False) 
            elif MODUS == "dev":
                slice_ = [_ for _ in range(100)]
                model = xgb.train(
                    params = params, 
                    dtrain = dtrain.slice(slice_),
                    num_boost_round = self.max_num_boost_round, 
                    evals = [(dval.slice(slice_),"val")], 
                    early_stopping_rounds = self.early_stopping_rounds,
                    verbose_eval=False) 
            
            # evaluate model on best iteration
            irange = (0, model.best_iteration)
            train_charrell = cindex(
                y_train["status"],
                 y_train["time"], 
                -model.predict(dtrain, iteration_range=irange))[0]
            val_charrell = cindex(
                y_val["status"], 
                y_val["time"], 
                -model.predict(dval, iteration_range=irange))[0]
            train_scores.append(train_charrell)
            val_scores.append(val_charrell)
        
        # compute mean and std of scores
        trial.set_user_attr("train_charrell_mean", np.nanmean(train_scores))
        trial.set_user_attr("train_charrell_std", np.nanstd(train_scores))
        trial.set_user_attr("val_charrell_mean", np.nanmean(val_scores))
        trial.set_user_attr("val_charrell_std", np.nanstd(val_scores))
        # log meta data
        trial.set_user_attr("ofold", self.outer_fold)
        # log best iteration
        trial.set_user_attr("best_iteration", model.best_iteration)

        return np.nanmean(val_scores)


#%% run study  =================================================================

# outer folds split
cv_outer = []
skf_outer = StratifiedKFold(n_splits=CV_NSPLITS)
for dev_idx, test_idx in skf_outer.split(X, y["time"]):
    cv_outer.append((dev_idx, test_idx))

run_name = f"leoss_{MODEL}_{ENDPOINT}"
with mlflow.start_run(run_name = run_name, run_id=args.run_id) as run:  # XXX
    mlflow.set_tags({"ENDPOINT":ENDPOINT,"MODEL":MODEL})

    # for metrics across outer folds
    dev_charrell_outer = []
    test_charrell_outer = []

    for i, outer in enumerate(cv_outer):

        if i not in args.ofold: 
            continue 

        print(f"\n=== outer fold {i} ===")
        # subset data wrt outer cross-validation
        dev_idx, test_idx = outer
        X_dev = X.iloc[dev_idx,:].copy()
        y_dev = y.iloc[dev_idx,:].copy()
        X_test = X.iloc[test_idx,:].copy()
        y_test = y.iloc[test_idx,:].copy()
        

        run_name_nstd = f"{run_name}_ofold{i}"
        with mlflow.start_run(run_name = run_name_nstd, nested=True) as run_nstd: 

            # inner fold indexing
            cv_inner =[]
            sfk_inner = StratifiedKFold(n_splits=CV_NSPLITS)        
            for train_idx, val_idx in sfk_inner.split(X_dev, y_dev["time"]):
                cv_inner.append((train_idx,val_idx))

            # hyperparameter tuning inner fold
            study_name = f"optstudy_{run_name_nstd}"
            storage = f"sqlite:///{study_name}.db"
            optuna_search = optuna.create_study(
                direction = "maximize",
                storage = storage,
                study_name = study_name,
                load_if_exists = True,
                )
            objective = Objective(
                X_dev, 
                y_dev,
                max_num_boost_round = XGB_MAX_NUM_BOOST_ROUND,
                cv = CV_NSPLITS, 
                early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
                outer_fold=i, 
                boosterparams=XGB_BOOSTERPARAMS)
            optuna_search.optimize(objective, OPTUNA_NTRIALS, n_jobs=OPTUNA_NJOBS)

            # save study
            fname = fr"./models/nCV/{run_name_nstd}_optuna_search.pkl"
            joblib.dump(optuna_search, fname)
            mlflow.log_artifact(fname)

            # refit to retrieve best model:
            ## preprocessing:
            ### one hot encoding because xgb does not support categorical variables
            enc = OneHotEncoder(sparse=False, drop="if_binary", categories = oh_all_categories)
            X_dev_oh = enc.fit_transform(X_dev)
            X_test_oh = enc.transform(X_test)
            X_dev_oh = pd.DataFrame(X_dev_oh, columns=enc.get_feature_names())
            X_test_oh = pd.DataFrame(X_test_oh, columns=enc.get_feature_names())
            ### make xgb data
            ddev = utils.general.xgb_make_survdmatrix(X_dev_oh, y_dev, label = "dev")
            dtest = utils.general.xgb_make_survdmatrix(X_test_oh, y_test, label = "test")
            ## refit model on best hyperparameters:            
            params = optuna_search.best_params
            best_iteration = optuna_search.best_trial._user_attrs["best_iteration"]
            params.update(XGB_BOOSTERPARAMS)                        
            if MODUS == "experiment":
                model = xgb.train(
                    params = params, 
                    dtrain = ddev, # XXX
                    num_boost_round = best_iteration,
                    verbose_eval=False) 
            elif MODUS == "dev":
                model = xgb.train(
                    params = params, 
                    dtrain = ddev.slice([_ for _ in range(100)]), # XXX
                    num_boost_round = best_iteration,
                    verbose_eval=False) 

            # evaluate model
            dev_charrell = cindex(y_dev["status"], y_dev["time"], -model.predict(ddev))[0]
            test_charrell = cindex(y_test["status"], y_test["time"], -model.predict(dtest))[0]
            dev_charrell_outer.append(dev_charrell)
            test_charrell_outer.append(test_charrell)

            # save optuna search
            fname = fr"./models/nCV/{run_name_nstd}_optuna_search.pkl"
            with open(fname, "wb") as f:
                pickle.dump(optuna_search,f)
            mlflow.log_artifact(fname)

            # save best model
            pipe = [
                ("enc",enc),
                ("model",model)
                ]
            fname_model = fr"./models/nCV/{run_name_nstd}.gzip.pkl"
            joblib.dump(pipe, fname_model, compress = "gzip")
            mlflow.log_artifact(fname_model)

            # log tags
            mlflow.set_tags({"ENDPOINT":ENDPOINT,"MODEL":MODEL,"ofold":i})
            # log params
            mlflow.log_params(params)
            # log metrics
            mlflow.log_metrics({"dev_charrell":dev_charrell, "test_charrell":test_charrell})

            # log save trials data frame               
            fname = fr"./reports/nCV/{run_name_nstd}_trials_dataframe.csv"
            trials_df = optuna_search.trials_dataframe()
            trials_df.columns = trials_df.columns.str.replace("test","val")
            # augment trials data frame
            trials_df["dev_charrell"] = dev_charrell
            trials_df["test_charrell"] = test_charrell
            trials_df["ofold"] = i
            trials_df["ENDPOINT"] = ENDPOINT
            trials_df["MODEL"] = MODEL
            trials_df.to_csv(fname, sep="\t")
            mlflow.log_artifact(fname)

            # save optuna plots
            fig = optuna.visualization.plot_optimization_history(optuna_search)
            fname = fr"./reports/nCV/{run_name_nstd}_optimization_history.html"
            fig.write_html(fname)
            mlflow.log_artifact(fname)

            fig = optuna.visualization.plot_param_importances(optuna_search)
            fname = fr"./reports/nCV/{run_name_nstd}_param_importances.html"
            fig.write_html(fname)
            mlflow.log_artifact(fname)

            fig = optuna.visualization.plot_slice(optuna_search)
            fname = fr"./reports/nCV/{run_name_nstd}_plot_sice.html"
            fig.write_html(fname)
            mlflow.log_artifact(fname)

        
        # outer fold summary metrics
        mlflow.log_metrics({
            "dev_charrell_mean":np.nanmean(dev_charrell_outer), 
            "dev_charrell_std":np.nanstd(dev_charrell_outer), 
            "test_charrell_mean":np.nanmean(test_charrell_outer),
            "test_charrell_std":np.nanstd(test_charrell_outer)
            })


