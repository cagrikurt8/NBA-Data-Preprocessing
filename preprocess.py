from datetime import datetime
import pandas as pd
import os
import requests
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector


# Check for ../Data directory presence
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# Download data if it is unavailable.
if 'nba2k-full.csv' not in os.listdir('../Data'):
    print('Train dataset loading.')
    url = "https://www.dropbox.com/s/wmgqf23ugn9sr3b/nba2k-full.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/nba2k-full.csv', 'wb').write(r.content)
    print('Loaded.')

data_path = "../Data/nba2k-full.csv"


# write your code here


def clean_data(path):
    dframe = pd.read_csv(path, parse_dates=[5, 10])
    dframe["team"].fillna("No Team", inplace=True)
    dframe["height"] = [i.split(" / ")[1] for i in dframe["height"]]
    dframe["weight"] = [i.split(" / ")[1].replace(" kg.", "") for i in dframe["weight"]]
    dframe["salary"] = [i.replace("$", "") for i in dframe["salary"]]
    dframe["height"] = dframe["height"].astype(float)
    dframe["weight"] = dframe["weight"].astype(float)
    dframe["salary"] = dframe["salary"].astype(float)
    dframe["country"] = [i if i == "USA" else "Not-USA" for i in dframe["country"]]
    dframe["draft_round"] = ["0" if i == "Undrafted" else i for i in dframe["draft_round"]]

    return dframe


def feature_data(dframe):
    dframe["version"] = [datetime.strptime(i[-2:], "%y") for i in dframe["version"]]
    dframe["age"] = [int(i.year) - int(k.year) for i, k in zip(dframe["version"], dframe["b_day"])]
    dframe["experience"] = [int(i.year) - int(k.year) for i, k in zip(dframe["version"], dframe["draft_year"])]
    dframe["bmi"] = [w / (h ** 2) for w, h in zip(dframe["weight"], dframe["height"])]
    dframe.drop(["version", "b_day", "draft_year", "weight", "height"], axis=1, inplace=True)

    for column in dframe:
        if len(dframe[column].value_counts()) >= 50 and dframe[column].dtype == object:
            dframe.drop(column, axis=1, inplace=True)

    return dframe


def multicol_data(dframe):
    numeric_columns = []

    for column in dframe.drop("salary", axis=1).columns:
        if dframe[column].dtype == np.int64 or dframe[column].dtype == np.float64:
            numeric_columns.append(column)

    corr_matrix = dframe.drop("salary", axis=1).corr(method='pearson')
    corr_pairs = []
    target = "salary"

    for idx in corr_matrix.index:
        for column in corr_matrix.columns:
            if (corr_matrix[idx][column] > 0.5 or corr_matrix[idx][column] < -0.5) and idx != column:
                corr_pairs.append([idx, column])

    for pair in corr_pairs:
        col1 = pair[0]
        col2 = pair[1]
        try:
            corr1 = dframe[target].corr(dframe[col1])
            corr2 = dframe[target].corr(dframe[col2])

            corr_dict = {corr1: col1, corr2: col2}

            less_column = corr_dict[min(corr1, corr2)]
            dframe.drop(less_column, axis=1, inplace=True)
        except KeyError:
            continue

    return dframe


def transform_data(dframe):
    X, y = dframe.drop("salary", axis=1), dframe["salary"]

    numeric_columns = [column if X[column].dtype != object else None for column in X.columns]
    categorical_columns = [column if X[column].dtype == object else None for column in X.columns]

    numeric_columns = list(filter(None, numeric_columns))
    categorical_columns = list(filter(None, categorical_columns))

    numeric_preprocessor = StandardScaler()
    categorical_preprocessor = OneHotEncoder(sparse=False)

    transformed_numeric = numeric_preprocessor.fit_transform(X[numeric_columns])
    transformed_categorical = categorical_preprocessor.fit_transform(X[categorical_columns])

    transformed_numeric_df = pd.DataFrame(columns=numeric_columns, data=transformed_numeric)

    cat_columns = list()

    for column_array in categorical_preprocessor.categories_:
        for column in column_array:
            cat_columns.append(column)

    transformed_categorical_df = pd.DataFrame(columns=cat_columns, data=transformed_categorical)
    X = pd.concat([transformed_numeric_df, transformed_categorical_df], axis=1)

    return X, y


X, y = transform_data(multicol_data(feature_data(clean_data(data_path))))

answer = {
    'shape': [X.shape, y.shape],
    'features': list(X.columns),
    }
print(answer)

print(X)
