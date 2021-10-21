import numpy as np
import pandas as pd
from pycox.evaluation import EvalSurv


def make_data_critical(
    data: pd.DataFrame, 
    X:pd.DataFrame, 
    t:int = 0
    ) -> 'tuple(pd.DataFrame, pd.DataFrame)':
    """create time-to-event endpoint for critical events

    columns defining critical events:
        CR_Symp_Delirium
        CR_Symp_DeliriumUnknown
        CR_Symp_Headache
        CR_Symp_HeadacheUnknown
        CR_Symp_Otherneuro
        CR_Symp_OtherNeuroUnknown
        CR_Compli_IntracerebBleed
        CR_Compli_IschemicStroke
        CR_Compli_CIM
        CR_Compli_CIP
        CR_Compli_Seizure

    Args:
        data (pd.DataFrame): leoss_decoded
        t (int): time threshold, default 0

    Returns:
        pd.DataFrame: time-to-event
    """    
    
    # event times critical events
    time_ = data["BL_CRstartdayNewCategories"].copy()
    # everything with a critical timestamp is defined as event (status = 1):
    status = (time_.notna()).astype(int)    
    follow_up = data["BL_ObservationalPeriodNewCat"].copy()
    # match follow-up to patients without critical events
    mask_censored = (status == 0) & (time_.isna()) 
    time_[mask_censored] = follow_up[mask_censored]
    # combine to data frame
    y = pd.DataFrame({"status" : status, "time" : time_})
    y.status = y.status.astype(bool)
    # remove observations with NA's in endpoint
    mask_notna = y.time.notna()
    X = X[mask_notna].copy()
    y = y[mask_notna]
    # coerce to int
    y.time = y.time.astype(int)
    # consider only patients with events after time >t
    mask_ge0 = y.time > t
    print(f"total observations = {y.shape[0]}, observations t>0 = {sum(mask_ge0)}")
    y = y[mask_ge0]
    X = X[mask_ge0]
    # print event rate
    n_total = y.shape[0]
    n_event = sum(y.status)
    per_event = round(n_event/n_total*100,1)
    print(f"endpoint critical events {n_event}/{n_total} = {per_event} %")

    return X, y


def make_data_complicated(
    data: pd.DataFrame, 
    X:pd.DataFrame, 
    t:int = 0
    ) -> 'tuple(pd.DataFrame, pd.DataFrame)':
    """create time-to-event endpoint for complicated events

    columns defining critical events:
        CR_Symp_Delirium
        CR_Symp_DeliriumUnknown
        CR_Symp_Headache
        CR_Symp_HeadacheUnknown
        CR_Symp_Otherneuro
        CR_Symp_OtherNeuroUnknown
        CR_Compli_IntracerebBleed
        CR_Compli_IschemicStroke
        CR_Compli_CIM
        CR_Compli_CIP
        CR_Compli_Seizure

    Args:
        data (pd.DataFrame): leoss_decoded
        t (int): time threshold, default 0

    Returns:
        pd.DataFrame: time-to-event
    """    
    
    # event times complicated events
    time_ = data["BL_COstartdayNewCategories"].copy()
    # everything with a complicated timestamp is defined as event (status = 1):
    status = (time_.notna()).astype(int)    
    follow_up = data["BL_ObservationalPeriodNewCat"].copy()
    # match follow-up to patients without complicated events
    mask_censored = (status == 0) & (time_.isna()) 
    time_[mask_censored] = follow_up[mask_censored]
    # combine to data frame
    y = pd.DataFrame({"status" : status, "time" : time_})
    y.status = y.status.astype(bool)
    # remove observations with NA's in endpoint
    mask_notna = y.time.notna()
    X = X[mask_notna].copy()
    y = y[mask_notna]
    # coerce to int
    y.time = y.time.astype(int)
    # consider only patients with events after time >t
    mask_ge0 = y.time > t
    print(f"total observations = {y.shape[0]}, observations t>0 = {sum(mask_ge0)}")
    y = y[mask_ge0]
    X = X[mask_ge0]
    # print event rate
    n_total = y.shape[0]
    n_event = sum(y.status)
    per_event = round(n_event/n_total*100,1)
    print(f"endpoint complicated events {n_event}/{n_total} = {per_event} %")

    return X, y


def make_data_death(
    data: pd.DataFrame, 
    X:pd.DataFrame, 
    t:int=0
    ) -> 'tuple(pd.DataFrame, pd.DataFrame)':

    """create time-to-event endpoint for death from cov19"""
    y = data[["BL_LastKnownStatus","BL_ObservationalPeriodNewCat"]].copy()
    y.rename(columns = {"BL_ObservationalPeriodNewCat" : "time"}, inplace=True)
    # set death flag
    y["status"] = 0
    mask_death = y["BL_LastKnownStatus"] == "Dead from COVID-19"
    y["status"][mask_death] = 1
    # type coercion        
    y.status = y.status.astype(bool)
    # remove observations with NA's in endpoint
    mask_notna = y.time.notna()
    X = X[mask_notna].copy()
    y = y[mask_notna]
    # coerce to int
    y.time = y.time.astype(int)
    # consider only patients with events after time >t
    mask_time = y.time > t
    y = y[mask_time]
    X = X[mask_time].copy()
    # print event rate
    n_total = y.shape[0]
    n_event = sum(y.status)
    per_event = round(n_event/n_total*100,1)
    print(f"endpoint death {n_event}/{n_total} = {per_event} %")
    # arrange columns in required order, drop BL_LastKnownStatus
    y = y[["status","time"]]
    return X, y


def filter_featurecols(X: pd.DataFrame, data_items: pd.DataFrame) -> pd.DataFrame:

    """preserve only columns in `X` which are marked as a feature in `data_items`"""
    feat_cols = data_items.iloc[:,0][data_items.feature == "x"].to_list()
    feat_cols = [col for col in feat_cols if col in X.columns.to_list()]
    Xf = X[feat_cols].copy()
    return Xf


def to_structured_array(tte:pd.DataFrame):

    """transform time-to-event pandas df into numpy structured array"""
    dt = tte.dtypes
    dtypes_ = list(zip(dt.index, dt))
    sa = np.array([tuple(_) for _ in tte.values], dtype=dtypes_)
    return sa


def get_pycox_scores(
    ytime: np.ndarray, 
    ystatus: np.ndarray,
    y_predict_survival_function: pd.DataFrame, 
    ytime_train = None
    ) -> dict:

    """compute scores using pycox library """
    if ytime_train is None:
        timegrid = np.unique(ytime)
    else:
        timegrid = np.unique(ytime_train)
    surv = pd.DataFrame(y_predict_survival_function.T).set_index(timegrid)
    ev = EvalSurv(surv, ytime, ystatus, censor_surv='km')
    ctd = ev.concordance_td('antolini')
    ibs = ev.integrated_brier_score(timegrid)
    inbll = ev.integrated_nbll(timegrid)
    return dict(ctd = ctd, ibs = ibs, inbll = inbll)


def xgb_make_survdmatrix(X:pd.DataFrame, y:pd.DataFrame, label:str=None):

    import xgboost as xgb
    # https://xgboost.readthedocs.io/en/latest/tutorials/aft_survival_analysis.html#how-to-use
    lb = y["time"].copy()
    ub = y["time"].astype(float).copy()
    ub[y["status"]==0] = +np.inf
    # remove special characters from feature names
    feature_names = X.columns
    scs = ["[","]",r"<"]
    for sc in scs:
        feature_names = feature_names.str.replace(sc,"")
    survdmat = xgb.DMatrix(
        X, 
        # label = label, 
        feature_names=feature_names, 
        label_lower_bound=lb,
        label_upper_bound=ub)
    return survdmat


def remove_special_chars(x:list):

    special = ["[","]",r"<"]
    x = pd.Series(x)
    for sc in special:
        x = x.str.replace(sc,"")
    return x.to_list()
    

def unify_ambiguous_missing(X: pd.DataFrame) -> pd.DataFrame:

    to_replace = [
        "Unknown","unknown", "None", "not done/unknown",
        "Not done/unknown", "Not determined/ Unknown"]
    for rpl in to_replace:
        X.replace(rpl,"missing",inplace=True)
    return X
