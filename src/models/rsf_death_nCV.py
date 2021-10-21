"""
DeepSurv nested 5x5-fold cross-validation
"""
 

# %% setup ========================================================================


import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import pickle
import optuna
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sksurv.ensemble import RandomSurvivalForest as RSF
from sksurv.metrics import concordance_index_censored as cindex
import utils.general
from utils.general import (
    make_data_death, 
    filter_featurecols, 
    unify_ambiguous_missing
    )

# Constants
RSF_NJOBS=30
RSF_NTREES=5000
CV_NSPLITS=5
RANDOM_STATE=42
OPTUNA_NTRIALS=15
# DATA = "critical"
ENDPOINT = "death"
MODEL = "rsf"


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

y = utils.general.to_structured_array(y)


#%% preprocessing ==============================================================


# outer folds split
cv_outer = []
skf_outer = StratifiedKFold(n_splits=CV_NSPLITS)
for dev_idx, test_idx in skf_outer.split(X, y["status"]):
    cv_outer.append((dev_idx, test_idx))

run_name = f"leoss_{MODEL}_{ENDPOINT}"
with mlflow.start_run(run_name = run_name) as run: 
    mlflow.set_tags({"ENDPOINT":ENDPOINT,"MODEL":MODEL})

    # for metrics across outer folds
    dev_charrell_outer = []
    test_charrell_outer = []

    for i, outer in enumerate(cv_outer):
        print(f"\n=== outer fold {i} ===")
        # slice data wrt outer folds
        dev_idx, test_idx = outer
        X_dev = X.iloc[dev_idx,:].copy()
        y_dev = y[dev_idx].copy()
        X_test = X.iloc[test_idx,:].copy()
        y_test = y[test_idx].copy()

        run_name_nstd = f"{run_name}_ofold{i}"
        with mlflow.start_run(run_name = run_name_nstd, nested=True) as run_nstd: 

            # hyperparameters
            params = {
                "rsf__max_depth" : optuna.distributions.IntUniformDistribution(8,64),
                }

            # model
            pipe = Pipeline([
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False)),
                ("rsf", RSF(n_estimators=RSF_NTREES, n_jobs=RSF_NJOBS))
                ])

            # inner fold indexing
            cv_inner =[]
            sfk_inner = StratifiedKFold(n_splits=CV_NSPLITS)        
            for train_idx, val_idx in sfk_inner.split(X_dev, y_dev["status"]):
                cv_inner.append((train_idx,val_idx))

            # hyperparameter tuning inner fold
            optuna_search = optuna.integration.OptunaSearchCV(
                estimator=pipe,
                param_distributions = params,
                cv = cv_inner,
                n_trials=OPTUNA_NTRIALS,    
                return_train_score = True,
                n_jobs = 1
                )
            optuna_search.fit(X_dev, y_dev)

            # save study
            fname = fr"./models/nCV/{run_name_nstd}_optuna_search.pkl"
            joblib.dump(optuna_search, fname)
            mlflow.log_artifact(fname)

            # evaluate best model
            yhat_dev = optuna_search.predict(X_dev)
            dev_charrell = cindex(y_dev["status"],y_dev["time"], yhat_dev)[0]
            yhat_test = optuna_search.predict(X_test)
            test_charrell = cindex(y_test["status"],y_test["time"], yhat_test)[0]
            dev_charrell_outer.append(dev_charrell)
            test_charrell_outer.append(test_charrell)

            # log tags
            mlflow.set_tags({"ENDPOINT":ENDPOINT,"MODEL":MODEL,"ofold":i})
            # log params
            mlflow.log_params(optuna_search.best_params_)
            # log metrics
            mlflow.log_metrics({"dev_charrell":dev_charrell, "test_charrell":test_charrell})

            # save trials data frame for convenience
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
            fig = optuna.visualization.plot_optimization_history(optuna_search.study_)
            fname = fr"./reports/nCV/{run_name_nstd}_optimization_history.html"
            fig.write_html(fname)
            mlflow.log_artifact(fname)        

        # outer fold summary metrics
        mlflow.log_metrics({
            "dev_charrell_mean":np.nanmean(dev_charrell_outer), 
            "dev_charrell_std":np.nanstd(dev_charrell_outer), 
            "test_charrell_mean":np.nanmean(test_charrell_outer),
            "test_charrell_std":np.nanstd(test_charrell_outer)
            })


