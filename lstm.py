
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import torch


def get_low_var_dummies(df, n):
    dummy_counts = df.loc[:, df.dtypes == np.uint8].apply(
                                                  lambda x: x[x == 1].count(),
                                                  axis=0)
    less_n_cols = dummy_counts[dummy_counts < len(df)*n]
    return less_n_cols


def col_cls_count(df, col):
        return df.groupby(col)[col].count()

def numeric_des(df):
    numeric_feat_data = df.loc[:, df.dtypes == 'float64']
    description = numeric_feat_data.describe(include='all').\
                append(numeric_feat_data.isnull().sum().rename('null val')).T
    return description
    
        

pd.DataFrame.get_low_var_dummies = get_low_var_dummies
pd.DataFrame.col_cls_count = col_cls_count
pd.DataFrame.numeric_des = numeric_des
# =============================================================================
# importing data
# =============================================================================

pd.set_option("display.max_columns", 20)
data = {
    'tra': pd.read_csv('../input/air_visit_data.csv'),
    'as': pd.read_csv('../input/air_store_info.csv'),
    'hs': pd.read_csv('../input/hpg_store_info.csv'),
    'ar': pd.read_csv('../input/air_reserve.csv'),
    'hr': pd.read_csv('../input/hpg_reserve.csv'),
    'id': pd.read_csv('../input/store_id_relation.csv'),
    'tes': pd.read_csv('../input/sample_submission.csv'),
    'hol': pd.read_csv('../input/date_info.csv'
                       ).rename(columns={'calendar_date': 'visit_date'})
    }

data['hr'] = pd.merge(data['hr'], data['id'], how="left", on="hpg_store_id")
for df in data:
    print(df)
    print(data[df].head(10))


# =============================================================================
# deal with date time in hpg_reserve and air_reserve table
# =============================================================================

for table in ["ar", "hr"]:
    # change date time variables into datetime object and to date
    for datetime in ["visit_datetime", "reserve_datetime"]:
        data[table][datetime] = pd.to_datetime(data[table][datetime]).dt.date

    # number of days people make reservations before the actual date
    data[table]["date_difference"] = (
                data[table]["visit_datetime"] - data[table]["reserve_datetime"]
                ).dt.days

    # group by id and dates and rename reserve visiors, since we need the ids
    # as our features, we set as_index = False
    data[table] = data[table].groupby(
            ['air_store_id', 'visit_datetime'], as_index=False)[
            "date_difference", "reserve_visitors"].sum().rename(
            columns={"visit_datetime": "visit_date"})


# =============================================================================
# deal with date time in train and test table
# =============================================================================

data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])

# split air id and dates for test table
data['tes']['air_store_id'] = data['tes']['id'].map(
                                        lambda x: "_".join(x.split("_")[:2]))
data['tes']['visit_date'] = data['tes']['id'].str.split('_', 2, True)[2]
data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])

# create new variables
# new_columns = column names, new_columns_action = attribute
new_columns = ["dow", "month", "visit_date"]
new_columns_action = ["dayofweek", "month", "date"]
for tables in ['tra', 'tes']:
    for index, value in enumerate(new_columns):
        data[tables][value] = getattr(data[tables]['visit_date'].dt,
                                      new_columns_action[index])


# =============================================================================
# gather information for stores in test set
# =============================================================================
unique_store = data["tes"].air_store_id.unique()
# make dataframe for store ids in test set with each day
store_info = [pd.DataFrame({"air_store_id": unique_store,
                            "dow": [i] * len(unique_store)}
                           )for i in range(7)]
store_info = pd.concat(store_info, axis=0, ignore_index=True)
# new columns names
new_columns = ['min_visitors', 'mean_visitors', 'median_visitors',
               'max_visitors', 'count_observations']

# corresponding action to add new columns
new_columns_action = ["min", "mean", "median", "max", "count"]

# aggregate
group = data['tra'].groupby(['air_store_id', 'dow'],
                            as_index=False)['visitors']

# add new columns
for index, value in enumerate(new_columns):
    column_ = getattr(group, new_columns_action[index])().rename(
                        columns={'visitors': new_columns[index]})
    store_info = pd.merge(store_info, column_, on=['air_store_id', 'dow'],
                          how='left')

# merge with air_store_info.csv
store_info = pd.merge(store_info, data['as'], on='air_store_id', how='left')
store_info.dtypes
len(data['as'])

# value count for each class 
air_genre_name_count = data['as'].col_cls_count('air_genre_name')
air_area_name_count = data['as'].col_cls_count('air_area_name')

# one hot encode
store_info = pd.get_dummies(store_info, columns=['air_genre_name',
                                                 'air_area_name'],
                            drop_first=True)

# check out statistics for numeric variables
numeric_feat_data = store_info.loc[:, store_info.dtypes == 'float64']
description = store_info.numeric_des() 

# find dummy variables that are less than n percent of the total counts
low_var_dummies = store_info.get_low_var_dummies(n =.01)

data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
data['hol'] = pd.get_dummies(data['hol'], columns=['day_of_week'],
                             drop_first=True)
data['hol']['visit_date'] = data['hol']['visit_date'].dt.date
train = pd.merge(data['tra'], data['hol'], how='left', on=['visit_date']) 
test = pd.merge(data['tes'], data['hol'], how='left', on=['visit_date']) 

train = pd.merge(data['tra'], store_info, how='left', on=['air_store_id','dow']) 
test = pd.merge(data['tes'], store_info, how='left', on=['air_store_id','dow'])

for df in ['ar','hr']:
    train = pd.merge(train, data[df], how='left', on=['air_store_id','visit_date']) 
    test = pd.merge(test, data[df], how='left', on=['air_store_id','visit_date'])
data
train = train.fillna(-1)
test = test.fillna(-1)
    # =============================================================================
# 
# =============================================================================
from sklearn.preprocessing import LabelEncoder
data = {
    'tra': pd.read_csv('../input/air_visit_data.csv'),
    'as': pd.read_csv('../input/air_store_info.csv'),
    'hs': pd.read_csv('../input/hpg_store_info.csv'),
    'ar': pd.read_csv('../input/air_reserve.csv'),
    'hr': pd.read_csv('../input/hpg_reserve.csv'),
    'id': pd.read_csv('../input/store_id_relation.csv'),
    'tes': pd.read_csv('../input/sample_submission.csv'),
    'hol': pd.read_csv('../input/date_info.csv').rename(columns={'calendar_date':'visit_date'})
    }

data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])
for df in ['ar','hr']:
    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])
    data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date
    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
    data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date    
    data[df]['reserve_datetime_diff'] = data[df].apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)
    data[df] = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].sum().rename(columns={'visit_datetime':'visit_date'})
data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])
data['tra']['dow'] = data['tra']['visit_date'].dt.dayofweek
data['tra']['year'] = data['tra']['visit_date'].dt.year
data['tra']['month'] = data['tra']['visit_date'].dt.month
data['tra']['visit_date'] = data['tra']['visit_date'].dt.date

data['tes']['visit_date'] = data['tes']['id'].map(lambda x: str(x).split('_')[2])
data['tes']['air_store_id'] = data['tes']['id'].map(lambda x: '_'.join(x.split('_')[:2]))
data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])
data['tes']['dow'] = data['tes']['visit_date'].dt.dayofweek
data['tes']['year'] = data['tes']['visit_date'].dt.year
data['tes']['month'] = data['tes']['visit_date'].dt.month
data['tes']['visit_date'] = data['tes']['visit_date'].dt.date
unique_stores = data['tes']['air_store_id'].unique()
stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': [i]*len(unique_stores)}) for i in range(7)], axis=0, ignore_index=True).reset_index(drop=True)

tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].min().rename(columns={'visitors':'min_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow']) 
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].mean().rename(columns={'visitors':'mean_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].median().rename(columns={'visitors':'median_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].max().rename(columns={'visitors':'max_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].count().rename(columns={'visitors':'count_observations'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])

train_des = train.describe(include='all').append(train.isnull().sum().rename('null_counts')).T

stores = pd.merge(stores, data['as'], how='left', on=['air_store_id']) 
lbl = LabelEncoder()
stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])

data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])
data['hol']['visit_date'] = data['hol']['visit_date'].dt.date
train = pd.merge(data['tra'], data['hol'], how='left', on=['visit_date']) 
test = pd.merge(data['tes'], data['hol'], how='left', on=['visit_date']) 