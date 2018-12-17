
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import torch


def get_low_var_dummies(df, n):
    dummy_counts = df.loc[:, df.dtypes == np.uint8].apply(
                                                  lambda x: x[x == 1].count(),
                                                  axis=0)
    less_n_cols = dummy_counts[dummy_counts <  len(df)*n]
    return less_n_cols


def col_cls_count(df, col):
        return df.groupby(col)[col].count()

# return descriptive dataframe of numeric variables in a dataframe


def numeric_des(df):
    numeric_feat_data = df.loc[:, df.dtypes == 'float64']
    description = numeric_feat_data.describe(
                  include='all').append(
                          numeric_feat_data.isnull(
                                          ).sum().rename('null val')).T
    return description


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # Input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # Forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # Put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # Drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def RMSLE(y, pred):
    return mean_squared_error(y, pred)**0.5

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

# aggregate data['tra']
# data['tra'] contains 'air_store_id', 'visit_date', 'visitors', 'dow', 'month'
group = data['tra'].groupby(['air_store_id', 'dow'],
                            as_index=False)['visitors']

# add new columns by applying functions onto column names
for index, value in enumerate(new_columns):
    column_ = getattr(group, new_columns_action[index])().rename(
                        columns={'visitors': new_columns[index]})

    # merge the list of new columns into a dataframe
    store_info = pd.merge(store_info, column_, on=['air_store_id',
                                                   'dow'],
                          how='left')

# merge with air_store_info.csv
# data['as'] contains :['air_store_id', 'air_genre_name', 'air_area_name',
# 'latitude', 'longitude']
store_info = pd.merge(store_info, data['as'], on='air_store_id',
                      how='left')
store_info.dtypes
len(data['as'])
store_info.columns
# 'air_store_id', 'dow', 'min_visitors', 'mean_visitors',
#     'median_visitors', 'max_visitors', 'count_observations',
#     'air_genre_name', 'air_area_name', 'latitude', 'longitude']

# value count for each class, I made this incase I wanna delete some
# of the categories
air_genre_name_count = data['as'].col_cls_count('air_genre_name')
air_area_name_count = data['as'].col_cls_count('air_area_name')

# one hot encode, but seems like orginal write used label encoder,
# i will find out which one is more practical
store_info = pd.get_dummies(store_info, columns=['air_genre_name',
                                                 'air_area_name'],
                            drop_first=True)

# check out statistics for numeric variables
numeric_feat_data = store_info.loc[:, store_info.dtypes == 'float64']
description = store_info.numeric_des()
# find dummy variables that are less than n percent of the total counts
# did not delete them for now, wish to see if the model could adjust it itself
low_var_dummies = store_info.get_low_var_dummies(n=.01)

# change all the dates into right data format and extract corresponding data
data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
data['hol'] = pd.get_dummies(data['hol'], columns=['day_of_week'],
                             drop_first=True)

# =============================================================================
# merge all df for train and test df
# =============================================================================
data['hol']['visit_date'] = data['hol']['visit_date'].dt.date
train = pd.merge(data['tra'], data['hol'], how='left',
                 on=['visit_date'])
test = pd.merge(data['tes'], data['hol'], how='left',
                on=['visit_date'])

data['tra'].columns
# 'air_store_id', 'visit_date', 'visitors', 'dow', 'month'
data['hol'].columns
# 'visit_date', 'holiday_flg', 'day_of_week_Monday',
#       'day_of_week_Saturday', 'day_of_week_Sunday', 'day_of_week_Thursday',
#       'day_of_week_Tuesday', 'day_of_week_Wednesday'

train = pd.merge(data['tra'], store_info, how='left',
                 on=['air_store_id', 'dow'])
test = pd.merge(data['tes'], store_info, how='left',
                on=['air_store_id', 'dow'])

for df in ['ar', 'hr']:
    train = pd.merge(train, data[df], how='left',
                     on=['air_store_id', 'visit_date'])
    test = pd.merge(test, data[df], how='left',
                    on=['air_store_id', 'visit_date'])

# =============================================================================
# train transformation
# =============================================================================
train = train.fillna(-1)
test = test.fillna(-1)
train.columns
train = train.sort_values('visit_date')
target_train = np.log1p(train.visitors.values)

col = [c for c in train if c not in ['air_store_id', 'visitors']]
train = train[col]
train.set_index('visit_date', inplace=True)
train['visitors'] = target_train
values = train.values
values = values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
col_len = scaled.shape[1]
reframed = series_to_supervised(scaled, 1, 1)
pd.set_option("display.max_columns", 30)
reframed.columns
reframed.head()
reframed.drop(reframed.columns[[i for i in range(col_len+1,258)]], axis=1, inplace=True)
reframed.head()
values = reframed.values
n_train_days = int(len(values) * 0.7)
train = values[:n_train_days, :]
test = values[n_train_days:, :]
# Split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# Reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
from keras.models import Sequential
from keras.layers import Dense, LSTM
multi_model = Sequential()
multi_model.add(LSTM(4, input_shape=(train_X.shape[1], train_X.shape[2])))
multi_model.add(Dense(1))
multi_model.compile(loss='mse', optimizer='adam')
multi_history = multi_model.fit(train_X, train_y, epochs=3,
                                batch_size=100, validation_data=(test_X, test_y),
                                verbose=1, shuffle=False)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# Invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# Invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
rmsle = RMSLE(inv_y, inv_yhat)
print('Test RMSLE: %.3f' % rmsle)