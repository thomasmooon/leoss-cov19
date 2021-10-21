from xgbse import XGBSEStackedWeibull


class my_XGBSE(XGBSEStackedWeibull):
    """wrapper to support hyperparameter optimization with the sklearn optuna API"""

    def __init__(
        self,
        xgb_objective = "survival:aft",
        xgb_eval_metric = "aft-nloglik",
        xgb_aft_loss_distribution = "normal",
        xgb_aft_loss_distribution_scale = 1,
        xgb_tree_method = "hist",
        xgb_learning_rate = 5e-2,
        xgb_max_depth = 8,
        xgb_booster = "dart",
        xgb_subsample = 0.5,
        xgb_min_child_weight = 50,
        xgb_colsample_bynode = 0.5,
        xgb_num_parallel_tree = 1,
        xgb_rate_drop = 0.0,
        wb_penalizer = 0.0,
        wb_l1_ratio = 0.0
        ):

        self.xgb_num_parallel_tree = xgb_num_parallel_tree
        self.xgb_rate_drop = xgb_rate_drop
        self.xgb_objective = xgb_objective
        self.xgb_eval_metric = xgb_eval_metric
        self.xgb_aft_loss_distribution = xgb_aft_loss_distribution
        self.xgb_aft_loss_distribution_scale = xgb_aft_loss_distribution_scale
        self.xgb_tree_method = xgb_tree_method
        self.xgb_learning_rate = xgb_learning_rate
        self.xgb_max_depth = xgb_max_depth
        self.xgb_booster = xgb_booster
        self.xgb_subsample = xgb_subsample
        self.xgb_min_child_weight = xgb_min_child_weight
        self.xgb_colsample_bynode = xgb_colsample_bynode
        self.wb_penalizer = wb_penalizer
        self.wb_l1_ratio = wb_l1_ratio
        
        xgb_params = {
            "objective" : xgb_objective,
            "eval_metric" : xgb_eval_metric,
            "aft_loss_distribution" : xgb_aft_loss_distribution,
            "aft_loss_distribution_scale" : xgb_aft_loss_distribution_scale,
            "tree_method" : xgb_tree_method,
            "learning_rate" : xgb_learning_rate,
            "max_depth" : xgb_max_depth,
            "booster" : xgb_booster,
            "subsample" : xgb_subsample,
            "min_child_weight" : xgb_min_child_weight,
            "colsample_bynode" : xgb_colsample_bynode,
            "num_parallel_tree" : xgb_num_parallel_tree,
            "rate_drop" : xgb_rate_drop
            }
        
        wb_params = {
            "penalizer" : wb_penalizer,
            "l1_ratio" : wb_l1_ratio
            }

        self.xgb_params = xgb_params
        self.weibull_params = wb_params
        self.persist_train = False
        self.feature_importances_ = None


# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import xgboost as xgb
from lifelines import WeibullAFTFitter
from xgbse.converters import convert_data_to_xgb_format, convert_y
from xgbse.non_parametric import get_time_bins
from sklearn.neighbors import BallTree

# support callback for xgb to enable pruning
# https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.XGBoostPruningCallback.html#optuna.integration.XGBoostPruningCallback
# https://github.com/optuna/optuna-examples/blob/main/xgboost/xgboost_integration.py


class XGBSEStackedWeibull_hyperband(XGBSEStackedWeibull):

    def fit(
        self,
        X,
        y,
        num_boost_round=1000,
        validation_data=None,
        early_stopping_rounds=None,
        verbose_eval=0,
        persist_train=False,
        index_id=None,
        time_bins=None,
        pruning_callback = None, # XXX
    ):
        """
        Fit XGBoost model to predict a value that is interpreted as a risk metric.
        Fit Weibull Regression model using risk metric as only independent variable.

        Args:
            X ([pd.DataFrame, np.array]): Features to be used while fitting XGBoost model

            y (structured array(numpy.bool_, numpy.number)): Binary event indicator as first field,
                and time of event or time of censoring as second field.

            num_boost_round (Int): Number of boosting iterations.

            validation_data (Tuple): Validation data in the format of a list of tuples [(X, y)]
                if user desires to use early stopping

            early_stopping_rounds (Int): Activates early stopping.
                Validation metric needs to improve at least once
                in every **early_stopping_rounds** round(s) to continue training.
                See xgboost.train documentation.

            verbose_eval ([Bool, Int]): Level of verbosity. See xgboost.train documentation.

            persist_train (Bool): Whether or not to persist training data to use explainability
                through prototypes

            index_id (pd.Index): User defined index if intended to use explainability
                through prototypes

            time_bins (np.array): Specified time windows to use when making survival predictions

        Returns:
            XGBSEStackedWeibull: Trained XGBSEStackedWeibull instance
        """

        E_train, T_train = convert_y(y)
        if time_bins is None:
            time_bins = get_time_bins(T_train, E_train)
        self.time_bins = time_bins

        # converting data to xgb format
        dtrain = convert_data_to_xgb_format(X, y, self.xgb_params["objective"])

        # converting validation data to xgb format
        evals = ()
        if validation_data:
            X_val, y_val = validation_data
            dvalid = convert_data_to_xgb_format(
                X_val, y_val, self.xgb_params["objective"]
            )
            evals = [(dvalid, "validation")]

        # training XGB
        self.bst = xgb.train(
            self.xgb_params,
            dtrain,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            evals=evals,
            verbose_eval=verbose_eval,
            callbacks = pruning_callback
        )
        self.feature_importances_ = self.bst.get_score()

        # predicting risk from XGBoost
        train_risk = self.bst.predict(dtrain, ntree_limit=self.bst.best_ntree_limit)

        # replacing 0 by minimum positive value in df
        # so Weibull can be fitted
        min_positive_value = T_train[T_train > 0].min()
        T_train = np.clip(T_train, min_positive_value, None)

        # creating df to use lifelines API
        weibull_train_df = pd.DataFrame(
            {"risk": train_risk, "duration": T_train, "event": E_train}
        )

        # fitting weibull aft
        self.weibull_aft = WeibullAFTFitter(**self.weibull_params)
        #self.weibull_aft.fit(weibull_train_df, "duration", "event", ancillary=True)
        self.weibull_aft.fit(weibull_train_df, "duration", "event")

        if persist_train:
            self.persist_train = True
            if index_id is None:
                index_id = X.index.copy()

            index_leaves = self.bst.predict(
                dtrain, pred_leaf=True, ntree_limit=self.bst.best_ntree_limit
            )
            self.tree = BallTree(index_leaves, metric="hamming")

        self.index_id = index_id

        return self
