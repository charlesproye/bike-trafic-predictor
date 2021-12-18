import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
import lightgbm as lgb


def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour

    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])


def _merge_external_data(X):
    file_path = Path(__file__).parent / 'external_data.csv'
    df_ext = pd.read_csv(file_path, parse_dates=['date'])
    
    X = X.copy()
    # When using merge_asof left frame need to be sorted
    X['orig_index'] = np.arange(X.shape[0])
    X = pd.merge_asof(X.sort_values('date'), df_ext[['date','t','u','td', 'rr24','tend24', 'confi', 'holiday']].sort_values('date'), on='date')
    # Sort back to the original order
    X = X.sort_values('orig_index')
    del X['orig_index']
    return X



def get_estimator():
    date_encoder = FunctionTransformer(_encode_dates)
    date_cols = ['month', 'day', 'weekday', 'hour', 'year']

    categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    categorical_cols = ["counter_name", "site_name"]
    numeric_cols = ['t','u', 'rr24','tend24']
    #period_cols = ['confi', 'holiday']

    preprocessor = ColumnTransformer(
        [
            ("date", OneHotEncoder(handle_unknown="ignore"), date_cols),
            ("cat", categorical_encoder, categorical_cols),
            ('numeric', StandardScaler(), numeric_cols),
            #('period', 'passthrough', period_cols)
        ]
    )

    params = {}
    params['learning_rate'] = 0.15
    params['boosting_type'] = 'gbdt'
    params['objective'] = 'regression'
    params['metric'] = ['rmse']
    params['num_leaves'] = 200
    params['min_data'] = 5
    params['max_depth'] = 30
    params['n_estimators'] = 500
    params['task']= 'train'

    regressor = lgb.LGBMRegressor(**params)

    pipe = make_pipeline(FunctionTransformer(_merge_external_data, validate=False), date_encoder, preprocessor, regressor)

    return pipe
