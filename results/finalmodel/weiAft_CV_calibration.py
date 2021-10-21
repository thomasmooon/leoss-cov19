"""
run on server for lifelines compatibility reasons
"""

# %% setup ========================================================================

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from lifelines import WeibullAFTFitter
import sys
sys.path.append("../../")
import utils.general

# %% load and filter data
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

# %% reload final model
fname_model = "/home/tlinden/leoss/mlruns/0/6b5ae4505f8a414489f73a34f7860f95/artifacts/leoss_weiAft_CV_death_CV.pkl.gzip"
pipe_ = joblib.load(fname_model)
enc = pipe_[0][1]
scaler = pipe_[1][1]
model = pipe_[2][1]

#%% parse feature names

names_map = dict(zip(X.columns, [f"x{i}_" for i,_ in enumerate(X.columns)]))
model_params = model.params_.reset_index()
for oldname, newname in names_map.items():
    model_params.covariate = model_params.covariate.str.replace(newname,oldname)

model_params.to_csv("weibull_aft_CV_finalmodel_params.csv")

# %% data preprocessing
Xt = pd.DataFrame(scaler.transform(enc.transform(X)))
Xt.columns = enc.get_feature_names()
Xy = pd.concat([ Xt.reset_index(drop=True), y.reset_index(drop=True)], axis=1)

# %% plot coefficients

# https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html?highlight=aft#plotting-the-coefficients
# matplotlib.pyplot.show()

plt.figure(figsize=(5,100))
ax = model.plot()
ax.get_figure().savefig("weiAft_CV_ImportanceCoeff.pdf")
plt.show()
#%% plot calibration 

# https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html?highlight=aft#model-probability-calibration
from lifelines.calibration import survival_probability_calibration

ax, _, _ = survival_probability_calibration(model, Xy, t0=7)
ax.get_figure().savefig("weiAft_CV_survival_probability_calibration_t7.pdf")
plt.show()

ax, _, _ = survival_probability_calibration(model, Xy, t0=14)
ax.get_figure().savefig("weiAft_CV_survival_probability_calibration_t14.pdf")
plt.show()

ax, _, _ = survival_probability_calibration(model, Xy, t0=21)
ax.get_figure().savefig("weiAft_CV_survival_probability_calibration_t21.pdf")
plt.show()

ax, _, _ = survival_probability_calibration(model, Xy, t0=28)
ax.get_figure().savefig("weiAft_CV_survival_probability_calibration_t28.pdf")
plt.show()

ax, _, _ = survival_probability_calibration(model, Xy, t0=35)
ax.get_figure().savefig("weiAft_CV_survival_probability_calibration_t35.pdf")
plt.show()

ax, _, _ = survival_probability_calibration(model, Xy, t0=80)
ax.get_figure().savefig("weiAft_CV_survival_probability_calibration_t80.pdf")
plt.show()

#%% panel plot


fig, axs= plt.subplots(nrows=2,ncols=3,sharex="col", figsize = (15,10)) # width, height
survival_probability_calibration(model, Xy, t0=7, ax=axs[0,0])
survival_probability_calibration(model, Xy, t0=14, ax=axs[0,1])
survival_probability_calibration(model, Xy, t0=21, ax=axs[0,2])
survival_probability_calibration(model, Xy, t0=28, ax=axs[1,0])
survival_probability_calibration(model, Xy, t0=35, ax=axs[1,1])
survival_probability_calibration(model, Xy, t0=80, ax=axs[1,2])
plt.tight_layout()
# plt.show()
plt.savefig("weiAft_finalmodel_survival_probability_calibration.pdf", bbox_inches="tight")
plt.close()
