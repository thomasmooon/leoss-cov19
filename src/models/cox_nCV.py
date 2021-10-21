"""
Cox-PH nested 5x5-fold cross-validation
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


# %% setup ====================================================================

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import pickle
import optuna
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from lifelines import CoxPHFitter, WeibullAFTFitter
from sksurv.metrics import concordance_index_censored as cindex
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
pdir = os.path.dirname(currentdir)
ppdir = os.path.dirname(pdir)
sys.path.append(ppdir)
import utils.general

# constants
CV_NSPLITS=5
RANDOM_STATE=42
OPTUNA_NTRIALS=2*20
OPTUNA_NJOBS=2
# DATA = "critical"
ENDPOINT = "death"
MODEL = "cox"

# %% load data ================================================================
data = pd.read_pickle("./data/processed/leoss_decoded.pkl", compression="zip")
data_items = pd.read_csv("./data/interim/data_items.txt", sep="\t")

#%% decide which data to be used further
filter_age = [
    "< 1 years", "4 - 8 years","1 - 3 years", "15 - 17 years", 
    "15-25 years", "> 85 years", "9 - 14 years"]
data = data[~data.BL_Age.isin(filter_age)]

X = utils.general.filter_featurecols(data, data_items)

# unify unknown / missingness
to_replace = [
    "Unknown","unknown", "None", "not done/unknown", 
    "Not done/unknown", "Not determined/ Unknown"]
for rpl in to_replace:
    X.replace(rpl,"missing",inplace=True)

# replace na's with missing, otherwise onehotencoder fails
X.fillna("missing", inplace=True)

# preprocess endpoint specific
if ENDPOINT == "critical":
    X, y = utils.general.make_data_critical(data, X)
elif ENDPOINT == "death":
    X, y = utils.general.make_data_death(data, X)

# get all possible categories and drop 'first' to avoid multi-colinearity issues
# during CPH training
oh_ = OneHotEncoder(drop = "first")
oh_.fit_transform(X)
oh_all_categories = oh_.categories_

#%% optuna objective function ==================================================
class Objective(object):
    def __init__(self, X, y, cv, outer_fold="NA", oh_categories="auto"):
        self.X = X
        self.y = y
        self.cv = cv
        self.outer_fold = outer_fold
        self.oh_categories = oh_categories

    def __call__(self, trial) -> float:
        X = self.X
        y = self.y

        # hyperparameters
        hparams = {
            "penalizer" : trial.suggest_loguniform("penalizer",1e-6, 1e-1),
            "l1_ratio" : trial.suggest_loguniform("l1_ratio",1e-1, 1),
            }              

        train_scores = []
        val_scores = []
        # one inner stratified cross validation loop
        for i, idx in enumerate(cv_inner):
            # slice data into folds
            train_idx, val_idx = idx
            X_train = X.iloc[train_idx,:]
            X_val = X.iloc[val_idx,:]
            y_train = y.iloc[train_idx,:]
            y_val = y.iloc[val_idx,:]         

            # preprocessing
            # one-hot encoding
            enc = OneHotEncoder(sparse=False, drop="first", categories = self.oh_categories)
            X_train_oh = enc.fit_transform(X_train)
            X_val_oh = enc.transform(X_val)
            # standardization
            scaler = StandardScaler()
            X_train_oh = scaler.fit_transform(X_train_oh)
            X_val_oh = scaler.transform(X_val_oh)
            # to df because required from CoxPHFitter, name columns
            X_train_oh = pd.DataFrame(X_train_oh)
            X_val_oh = pd.DataFrame(X_val_oh)
            X_train_oh.columns = enc.get_feature_names()
            X_val_oh.columns = enc.get_feature_names()
            # augment X with y as required, 
            # see https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html#penalties-and-sparse-regression
            X_train_oh.reset_index(drop=True, inplace=True)
            y_train.reset_index(drop=True, inplace=True)
            X_val_oh.reset_index(drop=True, inplace=True)
            y_val.reset_index(drop=True, inplace=True)
            X_train_oh_y = pd.concat([X_train_oh, y_train],axis=1)
            X_val_oh_y = pd.concat([X_val_oh, y_val],axis=1)

            # train models on hyperparameters
            try:
                # can fail dependent on hyperparameter
                cph = CoxPHFitter(
                    baseline_estimation_method="spline", 
                    n_baseline_knots=5, 
                    **hparams)
                cph.fit(X_train_oh_y, duration_col="time",event_col="status") # XXX
            except:
                print("\n trial failed")
                # https://optuna.readthedocs.io/en/stable/faq.html#how-are-nans-returned-by-trials-handled
                train_scores.append(np.nan)
                val_scores.append(np.nan)
                return np.nan 
            else:
                # evaluate model
                train_scores.append(cph.score(X_train_oh_y, scoring_method="concordance_index"))
                val_scores.append(cph.score(X_val_oh_y, scoring_method="concordance_index"))
        
        # compute mean and std of scores
        trial.set_user_attr("train_charrell_mean", np.nanmean(train_scores))
        trial.set_user_attr("train_charrell_std", np.nanstd(train_scores))
        trial.set_user_attr("val_charrell_mean", np.nanmean(val_scores))
        trial.set_user_attr("val_charrell_std", np.nanstd(val_scores))
        # log meta data
        trial.set_user_attr("ofold", self.outer_fold)
        
        return np.nanmean(val_scores)

#%% run study  =================================================================

# outer folds split
cv_outer = []
skf_outer = StratifiedKFold(n_splits=CV_NSPLITS)
for dev_idx, test_idx in skf_outer.split(X, y["status"]):
    cv_outer.append((dev_idx, test_idx))

run_name = f"leoss_{MODEL}_{ENDPOINT}"
with mlflow.start_run(run_name = run_name, run_id=args.run_id) as run:  # XXX
    mlflow.set_tags({"ENDPOINT":ENDPOINT,"MODEL":MODEL})

    # for metrics across outer folds
    dev_charrell_outer = []
    test_charrell_outer = []

    for i, outer in enumerate(cv_outer):

        if i not in args.ofold: # XXX
            continue 

        print(f"\n=== outer fold {i} ===")
        # slice data wrt outer folds
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
                direction = "maximize",
                storage = storage,
                study_name = study_name,
                load_if_exists = True,
                )
            optuna_search.optimize(
                Objective(X_dev, y_dev, cv_inner, i, oh_all_categories), 
                OPTUNA_NTRIALS)
            # save study
            fname = fr"./models/nCV/{run_name_nstd}_optuna_search.pkl"
            joblib.dump(optuna_search, fname)
            mlflow.log_artifact(fname)
            mlflow.log_artifact(study_name+".db")# XXX

            # refit to retrieve best model:
            ## preprocessing:
            ### one-hot encoding
            enc = OneHotEncoder(sparse=False, drop="first", categories = oh_all_categories)
            X_dev_oh = enc.fit_transform(X_dev)
            X_test_oh = enc.transform(X_test)
            ### standardization
            scaler = StandardScaler()
            X_dev_oh = scaler.fit_transform(X_dev_oh)
            X_test_oh = scaler.transform(X_test_oh)
            ### to df because required from CoxPHFitter, name columns
            X_dev_oh = pd.DataFrame(X_dev_oh)
            X_test_oh = pd.DataFrame(X_test_oh)
            X_dev_oh.columns = enc.get_feature_names()
            X_test_oh.columns = enc.get_feature_names()
            ### augment X with y as required, 
            #### see https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html#penalties-and-sparse-regression
            X_dev_oh.reset_index(drop=True, inplace=True)
            y_dev.reset_index(drop=True, inplace=True)
            X_test_oh.reset_index(drop=True, inplace=True)
            y_test.reset_index(drop=True, inplace=True)
            X_dev_oh_y = pd.concat([X_dev_oh, y_dev],axis=1)
            X_test_oh_y = pd.concat([X_test_oh, y_test],axis=1)
            ## refit model on best hyperparameters:
            # train models on hyperparameters
            try:
                best_params = optuna_search.best_params
                cph = CoxPHFitter(
                    baseline_estimation_method="spline", 
                    n_baseline_knots=5, 
                    **best_params)
                cph.fit(X_dev_oh_y, duration_col="time",event_col="status") # XXX
            except:
                print("\n trial failed")
                # https://optuna.readthedocs.io/en/stable/faq.html#how-are-nans-returned-by-trials-handled
                dev_charrell_outer.append(np.nan)
                test_charrell_outer.append(np.nan)
            else:
                # evaluate model
                dev_charrell = cph.score(X_dev_oh_y, scoring_method="concordance_index")
                test_charrell = cph.score(X_test_oh_y, scoring_method="concordance_index")
                dev_charrell_outer.append(dev_charrell)
                test_charrell_outer.append(test_charrell)

            # save best model # XXX
            pipe = [
                ("enc",enc),
                ("scaler",scaler),
                ("model",cph)
                ]
            fname_model = fr"./models/nCV/{run_name_nstd}.gzip.pkl"
            joblib.dump(pipe, fname_model, compress = "gzip")
            mlflow.log_artifact(fname_model)

            # log tags
            mlflow.set_tags({"ENDPOINT":ENDPOINT,"MODEL":MODEL,"ofold":i})
            # log params
            mlflow.log_params(best_params)
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

            # save optimization history
            fig = optuna.visualization.plot_optimization_history(optuna_search)
            fname = fr"./reports/nCV/{run_name_nstd}_optimization_history.html"
            fig.write_html(fname)
            mlflow.log_artifact(fname)

            fig = optuna.visualization.plot_param_importances(optuna_search)
            fname = fr"./reports/nCV/{run_name_nstd}_param_importances.html"
            fig.write_html(fname)
            mlflow.log_artifact(fname)
        
        # outer fold summary metrics
        mlflow.log_metrics({
            "dev_charrell_mean":np.nanmean(dev_charrell_outer), 
            "dev_charrell_std":np.nanstd(dev_charrell_outer), 
            "test_charrell_mean":np.nanmean(test_charrell_outer),
            "test_charrell_std":np.nanstd(test_charrell_outer)
            })


