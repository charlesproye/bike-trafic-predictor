import os
from os import getcwd
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import _name_estimators, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from pathlib import Path
from sklearn.model_selection import GridSearchCV
import numpy as np
import lightgbm as lgb


__file__ = Path("submissions") / "starting_kit" / "estimator.py"

def merge_external_data(X): 
    X = X.copy()
    X.loc[:, "date"] = pd.to_datetime(X["date"])
    
    file_path = Path(__file__).parent / "external_data.csv"
    df_weather = pd.read_csv(file_path, parse_dates=["date"])

    X_weather = df_weather[['date', 't', 'rr3']]

    X["orig_index"] = np.arange(X.shape[0])
    X_merged = pd.merge_asof(X.sort_values("date"), X_weather.sort_values("date"), on='date')
    X_merged['t'] = X_merged['t'].fillna(0)
    X_merged['rr3'] = X_merged['rr3'].fillna(0)
    X_merged = X_merged.sort_values("orig_index")
    del X_merged["orig_index"]
    
    return X_merged

def _encode_dates(X):
    X = X.copy()
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour
    X= X.drop(columns=["date"])
    return X


def get_estimator():
   
    # Call the merge_external_data function
    merge_external = FunctionTransformer(merge_external_data, validate=False)

    # Call the _encode_dates function to split the date column to several columns
    date_encoder = FunctionTransformer(_encode_dates)

    # Encode the final columns
    #date_one_encoder = OneHotEncoder(handle_unknown="ignore")
    #date_cols = ['year', 'month', 'day', 'weekday', 'hour']
      
    categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    categorical_cols = ["counter_name"]

    numeric_encoder = StandardScaler()
    numeric_cols = ['t', 'rr3', 'year', 'month', 'day', 'weekday', 'hour']

    preprocessor = ColumnTransformer(
        [
            #("date", date_one_encoder, date_cols),
            ("cat", categorical_encoder, categorical_cols),
            ("numeric", numeric_encoder, numeric_cols) 
        ]
    )

    params1 = {
        'lambda_l1' : 4, 
        'lambda_l2' : 18, 
        'num_leaves' : 99, 
        #'n_estimators' : 100, 
        'max_depth' : 11,
        'num_iterations' : 240,
        'min_child_samples' : 7,
        'max_bin' : 200, 
        #'subsample_for_bin' :  200,
        #'subsample' : 1,
        #'subsample_freq' : 1,
        'alpha' : 1, 
        #'reg_alpha' : 0.306, 
        #'reg_lambda' : 0.306, 
        #'min_split_gain' : 0.5,
        #'min_child_weight' : 1,
        #'learning_rate' : 0.025
        }

    params2 = {
        'max_depth': 24,
        'num_leaves': 2**10,
        'min_data_in_leaf':40,
        'learning_rate':0.1,
        'n_estimators': 130,
        'max_bin': 340,
        'lambda_l1': 0.01 ,
        'lambda_l2': 0,
        'min_gain_to_split':0,
        'boosting_type' : 'gbdt',
        'metric' : 'rmse'
        }

    regressor = lgb.LGBMRegressor(**params1, #random_state)

    pipe = make_pipeline(
        merge_external,
        date_encoder, 
        preprocessor, 
        regressor
    )
    

    # gros g

    return pipe