"""
DeepSurv nested 5x5-fold cross-validation
"""
 
# %% argparse =================================================================

import argparse
from math import isnan
parser = argparse.ArgumentParser()
parser.add_argument(
    "--ofold", 
    help="integer id of the outer fold allowed: [0,1,2,3,4]",
    type=int)
parser.add_argument(
    "--run_id", 
    help="mlflow parent run run-id", 
    type=str, 
    default=None)
args = parser.parse_args()


# %% setup ====================================================================

import mlflow
import numpy as np
import pandas as pd
import pickle, joblib
import optuna
#
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
#
import torch
from torch import nn
import torchtuples as tt
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
np.random.seed(1234)
_ = torch.manual_seed(123)
#
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
pdir = os.path.dirname(currentdir)
ppdir = os.path.dirname(pdir)
sys.path.append(ppdir)
from utils.general import make_data_death, make_data_critical, remove_special_chars, filter_featurecols, unify_ambiguous_missing

# %% functions
get_target = lambda df: (df['time'].values.astype("float32"), df['status'].values)

#%% constants

# MODUS = "dev"
MODUS = "experiment"
CV_NSPLITS=5
ENDPOINT = "death"
MODEL = "deepsurv"
RANDOM_STATE=42
OPTUNA_NTRIALS=20
OPTUNA_NJOBS=2
DS_MAXEPOCHS = 1000
DS_EARLY_STOPPING_ROUNDS = 10

# %% load data ====================================================================
data = pd.read_pickle("./data/processed/leoss_decoded.pkl", compression="zip")
data_items = pd.read_csv("./data/interim/data_items.txt", sep="\t")

filter_age = [
    "< 1 years", "4 - 8 years","1 - 3 years", "15 - 17 years", 
    "15-25 years", "> 85 years", "9 - 14 years"]
data = data[~data.BL_Age.isin(filter_age)]

#%% decide which data to be used further
X = filter_featurecols(data, data_items)

# unify unknown / missingness
X = unify_ambiguous_missing(X)

# replace na's with missing, otherwise onehotencoder fails
X.fillna("missing", inplace=True)

# preprocess endpoint specific
if ENDPOINT == "critical":
    X, y = make_data_critical(data, X)
elif ENDPOINT == "death":
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
        max_epochs=1000,
        cv=5,
        early_stopping_rounds = 10, 
        outer_fold="NA",
        oh_all_categories = None,
        device = None 
        ) -> float:

        self.X = X
        self.y = y
        self.max_epochs = max_epochs
        self.cv = cv
        self.early_stopping_rounds = early_stopping_rounds
        self.outer_fold = outer_fold
        self.oh_all_categories = oh_all_categories
        self.device=device


    def __call__(self, trial) -> float:
        X = self.X
        y = self.y
        
        # hyperparameters 
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1.0)
        l2 = trial.suggest_loguniform("l2", 1e-7, 1.0)
        dropout = trial.suggest_uniform("dropout",0, 0.5)
        batch_size = trial.suggest_categorical("batch_size",[32,64,512])
        act_fun = trial.suggest_categorical("act_fun", ["tanh","CELU"])
        initializer = trial.suggest_categorical("initializer",["kaiming_normal","orthogonal"])
        #
        n_feat = len(self.oh_all_categories)
        n_units_max = int(n_feat/2)
        n_units_step = int(n_feat/10)
        n_units_min = int(n_feat/20)
        n_units = trial.suggest_int("n_units",n_units_min,n_units_max,n_units_step)
        n_layers = trial.suggest_int("n_layers",1,10)
        num_nodes = [n_units] * n_layers,  
        #
        if initializer == "kaiming_normal":
            initializer_ = lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')
        elif initializer == "orthogonal":
            initializer_ = lambda w: nn.init.orthogonal_(w)
        #
        if act_fun == "tanh":
            act_fun_ = nn.Tanh
        elif act_fun == "CELU":
            act_fun_ = nn.CELU        

        # one iteration of the inner cross-validation
        train_scores = []
        val_scores = []        
        for i, idx in enumerate(cv_inner):
            
            # subset data wrt cv-fold
            train_idx, val_idx = idx
            X_train = X.iloc[train_idx,:]
            X_val = X.iloc[val_idx,:]
            y_train = y.iloc[train_idx,:]
            y_val = y.iloc[val_idx,:]

            # transform input
            ct = Pipeline([
                ("enc",OneHotEncoder(sparse=False, drop="first",categories=self.oh_all_categories)),
                ("scaler",StandardScaler())
                ])
            X_train_t = ct.fit_transform(X_train).astype("float32")
            X_val_t = ct.transform(X_val).astype("float32")
            cnames = remove_special_chars(ct["enc"].get_feature_names())           
            # convert y            
            y_train = get_target(y_train)
            y_val = get_target(y_val)       

            # train models on hyperparameters
            ## define model            
            net = tt.practical.MLPVanilla(
                in_features = X_train_t.shape[1], 
                num_nodes = num_nodes,              
                out_features = 1, 
                batch_norm = True,
                dropout = dropout,
                activation=act_fun_,
                w_init_= initializer_,
                output_bias=False)
            model = CoxPH(
                net=net, 
                optimizer=tt.optim.Adam(lr = learning_rate, weight_decay=l2),
                device=self.device)
            callbacks = [tt.callbacks.EarlyStopping()]            
            ## train model
            if MODUS == "experiment":
                log = model.fit(
                    input = X_train_t, 
                    target = y_train, 
                    batch_size = batch_size, 
                    epochs = self.max_epochs, 
                    callbacks=callbacks, 
                    verbose=False,
                    val_data=(X_val_t, y_val), 
                    )
            
            # evaluate model
            ## train set
            _ = model.compute_baseline_hazards()
            ev_train = EvalSurv(
                model.predict_surv_df(X_train_t), 
                y_train[0], 
                y_train[1], 
                censor_surv='km')            
            train_scores.append(ev_train.concordance_td())
            ## val set
            ev_val = EvalSurv(
                model.predict_surv_df(X_val_t), 
                y_val[0], 
                y_val[1], 
                censor_surv='km')            
            val_scores.append(ev_val.concordance_td())
        
        # log trial data
        ## mean and std of scores
        trial.set_user_attr("train_charrell_mean", np.nanmean(train_scores))
        trial.set_user_attr("train_charrell_std", np.nanstd(train_scores))
        trial.set_user_attr("val_charrell_mean", np.nanmean(val_scores))
        trial.set_user_attr("val_charrell_std", np.nanstd(val_scores))        
        ##  meta data
        trial.set_user_attr("ofold", self.outer_fold)
        ##  best iteration
        trial.set_user_attr("epochs", log.epoch)
        trial.set_user_attr("num_nodes", num_nodes)        

        return np.nanmean(val_scores)

#%% run study  =================================================================

# outer folds split
cv_outer = []
skf_outer = StratifiedKFold(n_splits=CV_NSPLITS)
for dev_idx, test_idx in skf_outer.split(X, y["status"]):
    cv_outer.append((dev_idx, test_idx))

run_name = f"leoss_{MODEL}_{ENDPOINT}"
with mlflow.start_run(run_name = run_name, run_id=args.run_id) as run: # XXX
# with mlflow.start_run(run_name = run_name) as run: 
    mlflow.set_tags({"ENDPOINT":ENDPOINT,"MODEL":MODEL})

    # for metrics across outer folds
    dev_charrell_outer = []
    test_charrell_outer = []

    for i, outer in enumerate(cv_outer):

        # skip fold until it matches the desired one
        
        # XXX
        if args.ofold != None:
            if i != args.ofold:            
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

            # inner fold optimization ------------------------------------------

            # inner fold indexing
            cv_inner =[]
            sfk_inner = StratifiedKFold(n_splits=CV_NSPLITS)        
            for train_idx, val_idx in sfk_inner.split(X_dev, y_dev["status"]):
                cv_inner.append((train_idx,val_idx))

            # hyperparameter tuning inner fold
            optuna_search = optuna.create_study(direction="maximize")
            objective = Objective(
                X_dev, 
                y_dev,
                max_epochs = DS_MAXEPOCHS,
                cv = CV_NSPLITS, 
                early_stopping_rounds=DS_EARLY_STOPPING_ROUNDS,
                outer_fold=i, 
                oh_all_categories=oh_all_categories,
                device=None # GPU
                )
            optuna_search.optimize(objective, OPTUNA_NTRIALS, n_jobs=OPTUNA_NJOBS)

            # save study
            fname = fr"./models/nCV/{run_name_nstd}_optuna_search.pkl"
            joblib.dump(optuna_search, fname)
            mlflow.log_artifact(fname)

            # refit ------------------------------------------------------------
            
            # transform input
            ct = Pipeline([
                ("enc",OneHotEncoder(sparse=False, drop="first",categories=oh_all_categories)),
                ("scaler",StandardScaler())
                ])
            X_dev_t = ct.fit_transform(X_dev).astype("float32")
            X_test_t = ct.transform(X_test).astype("float32")
            cnames = remove_special_chars(ct["enc"].get_feature_names())               
            # convert y            
            y_dev = get_target(y_dev)
            y_test = get_target(y_test)      

            # get best hyperparameters
            best_epochs = optuna_search.best_trial._user_attrs["epochs"]
            best_params = optuna_search.best_params.copy()
            num_nodes = optuna_search.best_trial._user_attrs["num_nodes"]
            #
            if best_params["initializer"] == "kaiming_normal":
                initializer = lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')
            elif best_params["initializer"] == "orthogonal":
                initializer = lambda w: nn.init.orthogonal_(w)
            #
            if best_params["act_fun"] == "tanh":
                act_fun = nn.Tanh
            elif best_params["act_fun"] == "CELU":
                act_fun = nn.CELU   

            ## define model            
            net = tt.practical.MLPVanilla(
                in_features = X_dev_t.shape[1], 
                num_nodes = num_nodes,              
                out_features = 1, 
                batch_norm = True,
                dropout = best_params["dropout"],
                activation = act_fun,
                w_init_= initializer,
                output_bias=False)
            model = CoxPH(
                net=net, 
                optimizer=tt.optim.Adam(
                    lr = best_params["learning_rate"], 
                    weight_decay=best_params["l2"]),
                    ) 

            ## train model
            if MODUS == "experiment":
                log = model.fit(
                    input = X_dev_t, 
                    target = y_dev, 
                    batch_size = best_params["batch_size"], 
                    epochs = best_epochs, 
                    verbose=False,
                    )
            
            # evaluate ----------------------------------------------------------
            _ = model.compute_baseline_hazards()
            ## train set
            ev_dev = EvalSurv(
                model.predict_surv_df(X_dev_t), 
                y_dev[0], 
                y_dev[1], 
                censor_surv='km')            
            dev_charrell_outer.append(ev_dev.concordance_td())
            ## test set
            ev_test = EvalSurv(
                model.predict_surv_df(X_test_t), 
                y_test[0], 
                y_test[1], 
                censor_surv='km')            
            test_charrell_outer.append(ev_test.concordance_td())

            # logging ----------------------------------------------------------

            # save optuna search
            fname = fr"./models/nCV/{run_name_nstd}_optuna_search.pkl"
            with open(fname, "wb") as f:
                pickle.dump(optuna_search,f)
            mlflow.log_artifact(fname)

            # save column transformer and best model             
            fname = fr"./models/nCV/{run_name_nstd}_bestmodel.pkl"
            transformations_and_model = (ct, model)
            joblib.dump(transformations_and_model,fname)            
            mlflow.log_artifact(fname)

            # log tags
            mlflow.set_tags({"ENDPOINT":ENDPOINT,"MODEL":MODEL,"ofold":i})
            # log params
            mlflow.log_params(optuna_search.best_params)
            # log metrics
            mlflow.log_metrics({
                "dev_charrell":ev_dev.concordance_td(), 
                "test_charrell":ev_test.concordance_td()})

            # log save trials data frame               
            fname = fr"./reports/nCV/{run_name_nstd}_trials_dataframe.csv"
            trials_df = optuna_search.trials_dataframe()
            trials_df.columns = trials_df.columns.str.replace("test","val")
            # augment trials data frame
            trials_df["dev_charrell"] = ev_dev.concordance_td()
            trials_df["test_charrell"] = ev_test.concordance_td()
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


