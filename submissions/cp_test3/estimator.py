from os import pread
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
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

def counters_done(X):
    mask1 = (data['counter_name']=='152 boulevard du Montparnasse E-O') & (data['date'] >= pd.to_datetime('2021/01/26')) & (data['date'] <= pd.to_datetime('2021/02/24'))
    mask1bis = (data['counter_name']=='152 boulevard du Montparnasse O-E') & (data['date'] >= pd.to_datetime('2021/01/26')) & (data['date'] <= pd.to_datetime('2021/02/24'))
    mask2 = (data['counter_name']=='20 Avenue de Clichy SE-NO') & (data['date'] >= pd.to_datetime('2021/05/06')) & (data['date'] <= pd.to_datetime('2021/07/21'))
    mask2bis = (data['counter_name']=='20 Avenue de Clichy NO-SE') & (data['date'] >= pd.to_datetime('2021/05/06')) & (data['date'] <= pd.to_datetime('2021/07/21'))
    X.drop(X[mask1].index, inplace=True)
    X.drop(X[mask1bis].index, inplace=True)
    X.drop(X[mask2].index, inplace=True)
    X.drop(X[mask2bis].index, inplace=True)
    return X

def confinement(X):
    date = pd.to_datetime(X['date'])
    X.loc[:, ['date_only']] = date
    new_date = [dt.date() for dt in X['date_only']]
    X.loc[:, ['date_only']] = new_date
    mask = ((X['date_only'] >= pd.to_datetime('2020/10/30'))
        & (X['date_only'] <= pd.to_datetime('2020/12/15'))
        | (X['date_only'] >= pd.to_datetime('2021/04/03'))
        & (X['date_only'] <= pd.to_datetime('2021/05/03')))
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
        (X['date_only'] >= pd.to_datetime('2020/12/15'))
        & (X['date_only'] < pd.to_datetime('2021/01/16'))
        & ((X['hour_only'] >= 20) | (X['hour_only'] <= 6))

        | 
        
        # Second curfew
        (X['date_only'] >= pd.to_datetime('2021/01/16'))
        & (X['date_only'] < pd.to_datetime('2021/03/20'))
        & ((X['hour_only'] >= 18) | (X['hour_only'] <= 6))

        |

        # Third curfew
        (X['date_only'] >= pd.to_datetime('2021/03/20'))
        & (X['date_only'] < pd.to_datetime('2021/05/19'))
        & ((X['hour_only'] >= 19) | (X['hour_only'] <= 6))

        |

        # Fourth curfew
        (X['date_only'] >= pd.to_datetime('2021/05/19'))
        & (X['date_only'] < pd.to_datetime('2021/06/9'))
        & ((X['hour_only'] >= 21) | (X['hour_only'] <= 6))

        |

        # Fifth curfew
        (X['date_only'] >= pd.to_datetime('2021/06/9'))
        & (X['date_only'] < pd.to_datetime('2021/06/20'))
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

    date_cols = ['month', 'day', 'weekday', 'hour', 'year']

    categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    categorical_cols = ["counter_name"]
    numeric_cols = ['t','u']
    period_cols = ['confi', 'curfew']

    preprocessor = ColumnTransformer(
        [
            ("date", OrdinalEncoder(), date_cols),
            ("cat", categorical_encoder, categorical_cols),
            ('numeric', StandardScaler(), numeric_cols),
            ('period', 'passthrough', period_cols)
        ]
    )

    paramsCh = {
        'max_depth': 24,
        'num_leaves': 2**10,
        'min_data_in_leaf':40,
        'learning_rate':0.1,
        'n_estimators': 130,
        'max_bin': 340,
        'lambda_l1': 0.01,
        'lambda_l2': 0,
        'min_gain_to_split':0,
        'boosting_type' : 'gbdt',
        'metric' : 'rmse'
    }

    paramsCam = {
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
        #'alpha' : 20,
        #'reg_alpha' : 0.306,
        #'reg_lambda' : 0.306,
        #'min_split_gain' : 0.5,
        #'min_child_weight' : 1,
        #'learning_rate' : 0.025
        }

    # Random state et gros gros parameter tuning et missing values avec un mask

    regressor = lgb.LGBMRegressor(**paramsCh, random_state=42)
    pipe = make_pipeline(FunctionTransformer(_merge_external_data, validate=False), confinement_encoder, curfew_encoder, date_encoder, preprocessor, regressor)

    return pipe
