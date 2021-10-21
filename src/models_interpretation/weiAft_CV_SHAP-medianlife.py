"""
estimate SHAP values based on prediction of median lifetimes
"""


# %% setup ========================================================================


import joblib
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lifelines import WeibullAFTFitter
import shap
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
pdir = os.path.dirname(currentdir)
ppdir = os.path.dirname(pdir)
sys.path.append(ppdir)
import utils.general

ENDPOINT = "death"
MODEL = "weiAft_CV"


# %% helpers ==================================================================


class Model_Shap():
    # wrap model components in class with .predict() method 
    # to adhere to shap.KernelExplainer() 

    def __init__(self, encoder, scaler, model) -> None:
        self.encoder = encoder
        self.scaler = scaler
        self.model = model       

    def predict(self, X):
        # transform
        X_t = self.encoder.transform(X)
        X_t = self.scaler.transform(X_t)
        X_t = pd.DataFrame(X_t, columns=self.encoder.get_feature_names())
        X_t.reset_index(drop=True, inplace=True)
        # predict
        return self.model.predict_percentile(X_t) # median lifetimes


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
    "Unknown","unknown", "None", "not done/unknown", "Not done/unknown",
    "Not determined/ Unknown"]
for rpl in to_replace:
    X.replace(rpl,"missing",inplace=True)

# replace na's with missing, otherwise onehotencoder fails
X.fillna("missing", inplace=True)

# preprocess endpoint specific
X, y = utils.general.make_data_death(data, X)

# get all possible categories and drop 'first' to avoid multi-colinearity issues
# during model training
oh_ = OneHotEncoder(drop = "first")
oh_.fit_transform(X)
oh_all_categories = oh_.categories_

# %% reload final model =======================================================

fname_model = "/home/tlinden/leoss/mlruns/0/6b5ae4505f8a414489f73a34f7860f95/artifacts/leoss_weiAft_CV_death_CV.pkl.gzip"
pipe_ = joblib.load(fname_model)
enc = pipe_[0][1]
scaler = pipe_[1][1]
model = pipe_[2][1]
model_shap = Model_Shap(enc,scaler,model)

# %% compute SHAP values ======================================================

with mlflow.start_run(run_name = "finalmodel_SHAP_medianlifetime") as run:

    mlflow.set_tags({
        "ENDPOINT":ENDPOINT,
        "MODEL":MODEL, 
        "explanation on": "median lifetime"
        })
    # train SHAP explainer, estimate SHAP values
    random_state = 42
    n = 1000
    X_sample = X.sample(n=n, random_state=random_state)
    explainer = shap.KernelExplainer(model_shap.predict, X_sample)
    shap_values = explainer.shap_values(X_sample)

    # save outcome
    print("saving files")
    fname_explainer = f"/home/tlinden/leoss/models/CV/leoss_weiAft_CV_SHAP-explainer-medianlifetime-n{n}.pkl.gz"
    joblib.dump(explainer, fname_explainer, compress=8)
    fname_shapvalues = f"/home/tlinden/leoss/models/CV/leoss_weiAft_CV_SHAP-shapvalues-medianlifetime-n{n}.pkl.gz"
    joblib.dump(shap_values, fname_shapvalues, compress=8)

    # mlflow log artifacts
    print("mlflow logging artifacts")
    mlflow.log_artifact(fname_explainer)
    mlflow.log_artifact(fname_shapvalues)

