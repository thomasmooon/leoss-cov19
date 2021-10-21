"""
XGBSE Stacked Weibull nested 5x5-fold cross-validation

"""

# %% argparse =================================================================


import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--ofold",
    help="integer id of the outer fold allowed: [0,1,2,3,4]", 
    type=int, 
    nargs="*", 
    default=[0,1,2,3,4])
parser.add_argument(
    "--run_id",
    help="mlflow parent run run-id",
    type=str,
    default=None)
args = parser.parse_args()
print(args)

# using argpase for `ofold` means, that only 1 outer fold is computed. hence
# no summary stats across all folds are possible
OUTER_FOLD_SUMMARY_STATS = [True if _ in args.ofold else False for _ in [0,1,2,3,4]]


# %% setup ====================================================================


import mlflow
import numpy as np
import pandas as pd
import pickle, joblib
import optuna
from pprint import pprint
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from xgbse.metrics import concordance_index as cindex
from xgbse.converters import convert_to_structured
from xgbse import XGBSEStackedWeibull

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
pdir = os.path.dirname(currentdir)
ppdir = os.path.dirname(pdir)
sys.path.append(ppdir)
from utils.general import (
    make_data_death, 
    remove_special_chars, 
    filter_featurecols, 
    unify_ambiguous_missing
    )
from utils.wrappers import XGBSEStackedWeibull_hyperband

#%% constants

# MODUS = "dev"
MODUS = "experiment"
CV_NSPLITS=5
ENDPOINT = "death"
MODEL = "xgbse_sWeiAft"
RANDOM_STATE=42
OPTUNA_NTRIALS=40
OPTUNA_NJOBS=2
# XGB_AFT_LOSS_DISTRIBUTION_SCALE has high impact on calibration, 
# see https://towardsdatascience.com/xgbse-improving-xgboost-for-survival-analysis-393d47f1384a
# not tuned here, because marginal effect on c-index. moreover xgb does only output
# a scalar (event time) and no risk-over-time. hence only suitable for ranking, hence
# poor calibration would NOT be a problem.
XGB_MAX_NUM_BOOST_ROUND = 1000
XGB_EARLY_STOPPING_ROUNDS = 10
XGB_NTHREAD = 1

# xgbse defaults from https://loft-br.github.io/xgboost-survival-embeddings/modules/stacked_weibull.html
XGB_DEFAULT_PARAMS = {
    "objective": "survival:aft",
    "eval_metric": "aft-nloglik",
    "aft_loss_distribution": "normal",
    "aft_loss_distribution_scale": 1,
    "tree_method": "hist",
    "learning_rate": 5e-2,
    "max_depth": 8,
    "booster": "dart",
    "subsample": 0.5,
    "min_child_weight": 50,
    "colsample_bynode": 0.5,
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
        oh_all_categories = None
        ) -> float:

        self.X = X
        self.y = y
        self.max_num_boost_round = max_num_boost_round
        self.cv = cv
        self.early_stopping_rounds = early_stopping_rounds
        self.nthread = nthread
        self.outer_fold = outer_fold
        self.oh_all_categories = oh_all_categories


    def __call__(self, trial) -> float:
        X = self.X
        y = self.y
        
        # xgb params

        ## default params
        xgb_params = {
            "objective": "survival:aft",
            "eval_metric": "aft-nloglik",
            "aft_loss_distribution": "normal",
            "aft_loss_distribution_scale": 1,
            "tree_method": "hist",
            "learning_rate": 5e-2,
            "max_depth": 8,
            "booster": "dart",
            "subsample": 0.5,
            "min_child_weight": 50,
            "colsample_bynode": 0.5,
            }

        ## hyper params
        xgb_hparams = {
            'learning_rate': trial.suggest_loguniform("learning_rate", 1e-1, 1.0),
            'max_depth': trial.suggest_int("max_depth", 2, 6),
            'subsample' : trial.suggest_uniform("subsample",0.5,1),
            'num_parallel_tree' : trial.suggest_int("num_parallel_tree", 2, 10), 
            "rate_drop" : trial.suggest_loguniform("rate_drop", 1e-3, 1e-1) 
            }        
        xgb_params.update(xgb_hparams)

        # aft params
        weibull_params = {
            "penalizer" : trial.suggest_loguniform("penalizer",1e-7, 1),
            "l1_ratio" : trial.suggest_loguniform("l1_ratio",1e-7, 1),
            }

        train_scores = []
        val_scores = []
        best_ntree_limit = []

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
            X_train_oh = pd.DataFrame(X_train_oh, columns=remove_special_chars(enc.get_feature_names()))
            X_val_oh = pd.DataFrame(X_val_oh, columns=remove_special_chars(enc.get_feature_names()))
            # convert y
            y_train = convert_to_structured(y_train["time"], y_train["status"])
            y_val = convert_to_structured(y_val["time"], y_val["status"])

            # train models on hyperparameters
            model = XGBSEStackedWeibull_hyperband(xgb_params=xgb_params, weibull_params=weibull_params)
            try:
                # AFT model can fit dependent on hyperparameters
                pruning_callback = optuna.integration.XGBoostPruningCallback(
                    trial, "validation-aft-nloglik")
                
                model = model.fit(   
                    X = X_train_oh, 
                    y = y_train,                 
                    num_boost_round = self.max_num_boost_round, 
                    validation_data = (X_val_oh, y_val), 
                    early_stopping_rounds = self.early_stopping_rounds,
                    verbose_eval=True,
                    pruning_callback = [pruning_callback]
                    ) 
            except:
                print("\n trial failed")
                # https://optuna.readthedocs.io/en/stable/faq.html#how-are-nans-returned-by-trials-handled
                train_scores.append(np.nan)
                val_scores.append(np.nan)
            else:
                # evaluate model
                train_scores.append(cindex(y_train, model.predict(X_train_oh)))
                val_scores.append(cindex(y_val, model.predict(X_val_oh)))
                best_ntree_limit.append(model.bst.best_ntree_limit)
        
        # compute mean and std of scores
        trial.set_user_attr("train_charrell_mean", np.nanmean(train_scores))
        trial.set_user_attr("train_charrell_std", np.nanstd(train_scores))
        trial.set_user_attr("val_charrell_mean", np.nanmean(val_scores))
        trial.set_user_attr("val_charrell_std", np.nanstd(val_scores))
        # log meta data
        trial.set_user_attr("ofold", self.outer_fold)
        # log best iteration
        # log best iteration
        trial.set_user_attr("best_ntree_limit_mean", np.nanmean(best_ntree_limit))
        trial.set_user_attr("best_ntree_limit_std", np.nanstd(best_ntree_limit))

        return -np.nanmean(val_scores)


#%% run study  =================================================================


# outer folds split
cv_outer = []
skf_outer = StratifiedKFold(n_splits=CV_NSPLITS)
for dev_idx, test_idx in skf_outer.split(X, y["status"]):
    cv_outer.append((dev_idx, test_idx))

run_name = f"leoss_{MODEL}_{ENDPOINT}"
with mlflow.start_run(run_name = run_name, run_id=args.run_id) as run: 
    mlflow.set_tags({"ENDPOINT":ENDPOINT,"MODEL":MODEL})

    # for metrics across outer folds
    dev_charrell_outer = []
    test_charrell_outer = []

    for i, outer in enumerate(cv_outer):

        # skip fold until it matches the desired one
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
            for train_idx, val_idx in sfk_inner.split(X_dev, y_dev["status"]):
                cv_inner.append((train_idx,val_idx))

            # hyperparameter tuning inner fold
            study_name = f"optstudy_{run_name_nstd}"# XXX
            storage = f"sqlite:///{study_name}.db"

            optuna_search = optuna.create_study(
                direction = "minimize",
                storage = storage,
                study_name = study_name,
                load_if_exists = True, 
                pruner=optuna.pruners.HyperbandPruner()
                )

            objective = Objective(
                X_dev, 
                y_dev,
                max_num_boost_round = XGB_MAX_NUM_BOOST_ROUND,
                cv = CV_NSPLITS, 
                early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
                outer_fold=i, 
                oh_all_categories=oh_all_categories)

            optuna_search.optimize(objective, OPTUNA_NTRIALS, n_jobs=OPTUNA_NJOBS)

            # save study
            fname = fr"./models/nCV/{run_name_nstd}_optuna_search.pkl"
            joblib.dump(optuna_search, fname)
            mlflow.log_artifact(fname)
            mlflow.log_artifact(study_name+".db")# XXX

            # refit to retrieve best model:
            
            ## preprocessing:        

            ### one hot encoding because xgb does not support categorical variables
            enc = OneHotEncoder(sparse=False, drop="if_binary", categories = oh_all_categories)
            X_dev_oh = enc.fit_transform(X_dev)
            X_test_oh = enc.transform(X_test)
            X_dev_oh = pd.DataFrame(X_dev_oh, columns=remove_special_chars(enc.get_feature_names()))
            X_test_oh = pd.DataFrame(X_test_oh, columns=remove_special_chars(enc.get_feature_names()))
            
            ### convert y
            y_dev = convert_to_structured(y_dev["time"], y_dev["status"])
            y_test = convert_to_structured(y_test["time"], y_test["status"])            
            
            ## get best hyperparameters

            ### general params
            best_iteration = int(optuna_search.best_trial._user_attrs["best_ntree_limit_mean"])
            
            ### weibull params
            best_params = optuna_search.best_params.copy()
            weibull_params = {
                "penalizer" : best_params.pop("penalizer"),
                "l1_ratio" : best_params.pop("l1_ratio"),
                }
            
            ### xgb params
            xgb_params = XGB_DEFAULT_PARAMS.copy()
            xgb_params.update(best_params)
            
            ## refit -- can fail dependent on hyperparameter            
            model = XGBSEStackedWeibull(xgb_params=xgb_params, weibull_params=weibull_params)
            try:
                model = model.fit(   
                    X = X_dev_oh, 
                    y = y_dev,                 
                    num_boost_round = best_iteration, 
                    verbose_eval=False) 
            except:
                print("\n fit failed")
                # https://optuna.readthedocs.io/en/stable/faq.html#how-are-nans-returned-by-trials-handled
                dev_charrell_outer.append(np.nan)
                test_charrell_outer.append(np.nan)
            else:
                # evaluate model
                dev_charrell = cindex(y_dev, model.predict(X_dev_oh))
                test_charrell = cindex(y_test, model.predict(X_test_oh))
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
            mlflow.log_params(optuna_search.best_params)
            # log metrics
            mlflow.log_metrics({
                "dev_charrell":dev_charrell, 
                "test_charrell":test_charrell})

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

            fig = optuna.visualization.plot_intermediate_values(optuna_search)
            fname = fr"./reports/nCV/{run_name_nstd}_plot_intermediate.html"
            fig.write_html(fname)
            mlflow.log_artifact(fname)

        
        # outer fold summary metrics
        if OUTER_FOLD_SUMMARY_STATS:
            mlflow.log_metrics({
                "dev_charrell_mean":np.nanmean(dev_charrell_outer), 
                "dev_charrell_std":np.nanstd(dev_charrell_outer), 
                "test_charrell_mean":np.nanmean(test_charrell_outer),
                "test_charrell_std":np.nanstd(test_charrell_outer)
                })


