"""
Weibull AFT 
- compute best parameters using 5xCV
- Train final model on all data
"""
 

# %% setup ========================================================================


import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import joblib
import optuna
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from lifelines import WeibullAFTFitter
from sksurv.metrics import concordance_index_censored as cindex
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
pdir = os.path.dirname(currentdir)
ppdir = os.path.dirname(pdir)
sys.path.append(ppdir)
from utils.general import (
    make_data_death,
    filter_featurecols, 
    unify_ambiguous_missing
    )

# MODUS = "dev"
MODUS = "experiment"
CV_NSPLITS=5
RANDOM_STATE=42
OPTUNA_NTRIALS=40
OPTUNA_NJOBS=2
ENDPOINT = "death"
MODEL = "weiAft_CV"


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

# get all possible categories and drop 'first' to avoid multi-colinearity issues
# during model training
oh_ = OneHotEncoder(drop = "first")
oh_.fit_transform(X)
oh_all_categories = oh_.categories_


#%% optuna objective function ==================================================


class Objective(object):
    def __init__(self, X, y, cv, oh_categories="auto"):
        self.X = X
        self.y = y
        self.cv = cv
        self.oh_categories = oh_categories

    def __call__(self, trial) -> float:
        X = self.X
        y = self.y

        # hyperparameters
        hparams = {
            "penalizer" : trial.suggest_loguniform("penalizer",1e-1, 1),
            "l1_ratio" : trial.suggest_loguniform("l1_ratio",1e-6, 1e-2),
            }              

        train_scores = []
        val_scores = []
        # one inner stratified cross validation loop
        for i, idx in enumerate(self.cv):
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
            # to df because required from WeibullAFTFitter, name columns
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
                model = WeibullAFTFitter(**hparams)
                if MODUS == "experiment":
                    model.fit(X_train_oh_y, duration_col="time",event_col="status")
                elif MODUS == "dev":
                    model.fit(X_train_oh_y.iloc[:100,-50:], duration_col="time",event_col="status")
            except:
                print("\n trial failed")
                # https://optuna.readthedocs.io/en/stable/faq.html#how-are-nans-returned-by-trials-handled
                train_scores.append(np.nan)
                val_scores.append(np.nan)
            else:
                # evaluate model
                train_scores.append(model.score(X_train_oh_y, scoring_method="concordance_index"))
                val_scores.append(model.score(X_val_oh_y, scoring_method="concordance_index"))
        
        # compute mean and std of scores
        trial.set_user_attr("train_charrell_mean", np.nanmean(train_scores))
        trial.set_user_attr("train_charrell_std", np.nanstd(train_scores))
        trial.set_user_attr("val_charrell_mean", np.nanmean(val_scores))
        trial.set_user_attr("val_charrell_std", np.nanstd(val_scores))
        
        return np.nanmean(val_scores)


#%% run study  =================================================================

# outer folds split
cv = []
skf_outer = StratifiedKFold(n_splits=CV_NSPLITS)
for train_idx, val_idx in skf_outer.split(X, y["status"]):
    cv.append((train_idx, val_idx))

run_name = f"leoss_{MODEL}_{ENDPOINT}_CV"
with mlflow.start_run(run_name = run_name) as run:                 
    
    # tune ---------------------------------------------------------------------

    study_name = f"optstudy_{run_name}" # XXX
    storage = f"sqlite:///{study_name}.db"
    study = optuna.create_study(
        direction = "maximize",
        storage = storage,
        study_name = study_name,
        load_if_exists = True, # XXX
        )
    study.optimize(
        Objective(X, y, cv, oh_all_categories), 
        OPTUNA_NTRIALS)    

    # refit to retrieve final model --------------------------------------------
    
    # preprocessing
    enc = OneHotEncoder(sparse=False, drop="first", categories = oh_all_categories)
    scaler = StandardScaler()
    Xt = pd.DataFrame(scaler.fit_transform(enc.fit_transform(X)))
    Xt.columns = enc.get_feature_names()
    
    # augment X with y as required from method, see 
    # https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html#penalties-and-sparse-regression  
    Xy = pd.concat([
        Xt.reset_index(drop=True), y.reset_index(drop=True)],
        axis=1)
    
    # refit model on best hyperparameters    
    try:
        best_params = study.best_params
        model = WeibullAFTFitter(**best_params)
        if MODUS == "experiment":
            model.fit(Xy, duration_col="time",event_col="status")
        elif MODUS == "dev":
            model.fit(Xy.iloc[:100,-50:], duration_col="time",event_col="status")
    except:
        print("\n fit failed ")
    else:
        # evaluate model
        final_charrell = model.score(Xy, scoring_method="concordance_index")

    # logging ------------------------------------------------------------------
    
    # save study
    fname = fr"./models/CV/{run_name}_study.pkl"
    joblib.dump(study, fname)
    mlflow.log_artifact(fname)
    mlflow.log_artifact(study_name+".db")
    
    # save best model
    pipe = [
        ("enc",enc),
        ("scaler",scaler),
        ("model",model)
        ]
    fname_model = fr"./models/CV/{run_name}.pkl.gzip"
    joblib.dump(pipe, fname_model, compress = "gzip")
    mlflow.log_artifact(fname_model)

    # log tags
    mlflow.set_tags({"ENDPOINT":ENDPOINT,"MODEL":MODEL})
    # log params
    mlflow.log_params(best_params)
    # log metrics
    mlflow.log_metrics({"final_charrell":final_charrell})

    # log save trials data frame               
    fname = fr"./reports/CV/{run_name}_trials_dataframe.csv"
    trials_df = study.trials_dataframe()
    # augment trials data frame
    trials_df["final_charrell"] = final_charrell
    trials_df["ENDPOINT"] = ENDPOINT
    trials_df["MODEL"] = MODEL
    trials_df.to_csv(fname, sep="\t")
    mlflow.log_artifact(fname)

    # save optimization history
    fig = optuna.visualization.plot_optimization_history(study)
    fname = fr"./reports/CV/{run_name}_optimization_history.html"
    fig.write_html(fname)
    mlflow.log_artifact(fname)
    #
    fig = optuna.visualization.plot_param_importances(study)
    fname = fr"./reports/CV/{run_name}_param_importances.html"
    fig.write_html(fname)
    mlflow.log_artifact(fname)
    #
    fig = optuna.visualization.plot_slice(study)
    fname = fr"./reports/nCV/{run_name}_plot_sice.html"
    fig.write_html(fname)
    mlflow.log_artifact(fname)
        
        


