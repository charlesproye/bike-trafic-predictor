from os import pread
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
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
 
 
def confinement(X):
    date = pd.to_datetime(X['date'])
    X.loc[:, ['date_only']] = date
    new_date = [dt.date() for dt in X['date_only']]
    X.loc[:, ['date_only']] = new_date
    mask = ((X['date_only'] >= pd.to_datetime('2020/10/30').date())
        & (X['date_only'] <= pd.to_datetime('2020/12/15').date())
        | (X['date_only'] >= pd.to_datetime('2021/04/03').date())
        & (X['date_only'] <= pd.to_datetime('2021/05/03').date()))
    X['confi'] = np.where(mask, 1, 0)
    return X
 
def curfew(X):
    date = pd.to_datetime(X['date'])
    X.loc[:, ['date_only']] = date
    new_date = [dt.date() for dt in X['date_only']]
    X.loc[:, ['date_only']] = new_date
    X.loc[:, ['hour_only']] = date
    new_hour = [dt.hour for dt in X['hour_only']]
    X.loc[:, ['hour_only']] = new_hour
    mask = (
        #First curfew
        (X['date_only'] >= pd.to_datetime('2020/12/15').date())
        & (X['date_only'] < pd.to_datetime('2021/01/16').date())
        & ((X['hour_only'] >= 20) | (X['hour_only'] <= 6))
 
        | 
        
        # Second curfew
        (X['date_only'] >= pd.to_datetime('2021/01/16').date())
        & (X['date_only'] < pd.to_datetime('2021/03/20').date())
        & ((X['hour_only'] >= 18) | (X['hour_only'] <= 6))
 
        |
 
        # Third curfew
        (X['date_only'] >= pd.to_datetime('2021/03/20').date())
        & (X['date_only'] < pd.to_datetime('2021/05/19').date())
        & ((X['hour_only'] >= 19) | (X['hour_only'] <= 6))
 
        |
 
        # Fourth curfew
        (X['date_only'] >= pd.to_datetime('2021/05/19').date())
        & (X['date_only'] < pd.to_datetime('2021/06/9').date())
        & ((X['hour_only'] >= 21) | (X['hour_only'] <= 6))
 
        |
 
        # Fifth curfew
        (X['date_only'] >= pd.to_datetime('2021/06/9').date())
        & (X['date_only'] < pd.to_datetime('2021/06/20').date())
        & ((X['hour_only'] >= 21) | (X['hour_only'] <= 6))
        )
    X['curfew'] = np.where(mask, 1, 0)
 
    return X.drop(columns=['hour_only', 'date_only'])
 
 
def _merge_external_data(X):
    file_path = Path(__file__).parent / 'external_data.csv'
    df_ext = pd.read_csv(file_path, parse_dates=['date'])
    X = X.copy()
    # When using merge_asof left frame need to be sorted
    X['orig_index'] = np.arange(X.shape[0])
    X = pd.merge_asof(X.sort_values('date'), df_ext[['date','t','u', 'rr3']].sort_values('date'), on='date')
    # Sort back to the original order
    X = X.sort_values('orig_index')
    del X['orig_index']
    return X
 
 
 
def get_estimator():
    confinement_encoder = FunctionTransformer(confinement)
    curfew_encoder = FunctionTransformer(curfew)
    date_encoder = FunctionTransformer(_encode_dates)
    merge = FunctionTransformer(_merge_external_data, validate=False)
 
    date_cols = ['month', 'day', 'weekday', 'year']
 
    categorical_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=100)
    categorical_cols = ["counter_name"]
    numeric_cols = ['t','u']
    period_cols = ['confi', 'curfew']
    hour_col = ['hour']
 
    preprocessor = ColumnTransformer(
        [
            ("date", StandardScaler(), date_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ('numeric', StandardScaler(), numeric_cols),
            ('period', 'passthrough', period_cols),
            ('hour', PolynomialFeatures(degree=2), hour_col)
        ]
    )


    regressor = lgb.LGBMRegressor(random_state=21)
    pipe = make_pipeline(merge, confinement_encoder, curfew_encoder, date_encoder, preprocessor, regressor)
 
    return pipe
 