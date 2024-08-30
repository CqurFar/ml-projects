# Libraries
import os
import optuna
import numpy as np
import pandas as pd
from janitor import clean_names
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score


# Others
# import lightgbm as lgb
# import category_encoders as ce
# import matplotlib.pyplot as plt
# from statsmodels.tsa.seasonal import STL
# from joblib import Parallel, delayed, dump, load
# from sklearn.model_selection import KFold, RandomizedSearchCV, GridSearchCV




# Import csv
def import_csv(directory="data"):
    files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(".csv")]
    data_dict = {}
    na_counts = {}

    for file in files:
        df = clean_names(pd.read_csv(file, na_values=["N/A", "NA", "", "."]))
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        data_name = os.path.splitext(os.path.basename(file))[0]
        data_dict[data_name] = df
        na_counts[data_name] = df.isna().sum().sum()

    na_df = pd.DataFrame(list(na_counts.items()), columns=["dataset", "na_count"])
    globals().update(data_dict)
    return na_df

import_info = import_csv("Store Sales/sales_data")


# Standardization of dates
def align_dates(df, reference_dates):
    all_dates = reference_dates["date"]
    df_aligned = df.set_index("date").reindex(all_dates).rename_axis("date").reset_index()
    return df_aligned

test["data_type"] = "test"
train["data_type"] = "train"
train_test = pd.concat([test, train], ignore_index=True)

train_test["weekday"] = train_test["date"].dt.day_name()
train_test["year_mo"] = train_test["date"].dt.to_period("M").astype(str)

train_test_dates = train_test[["date"]].drop_duplicates()
dates = pd.DataFrame({"date": pd.date_range(start="2013-01-01", end="2017-08-31", freq="D")})
dates_seq = dates[dates["date"].isin(train_test_dates["date"])]


# oil df
oil_seq = align_dates(oil, dates_seq).rename(columns={"dcoilwtico": "oil_price"})

def interpolate_price(df, column_name="oil_price"):
    df[column_name] = df[column_name].interpolate(method="pchip")
    df[column_name] = df[column_name].bfill().ffill()
    df[column_name] = df[column_name].round(2)
    return df

oil_price = interpolate_price(oil_seq)


# holidays df
def filtering_holidays(df):
    df["type"] = df["type"].replace({"Bridge": "Additional", "Transfer": "Additional"})
    df = df[~((df["transferred"] == True) | (df["type"] == "Work Day"))]
    df = df.drop(columns=["transferred", "description", "locale_name"])
    df = df.rename(columns={"type": "day_type"})
    return df

holidays_fil = filtering_holidays(holidays_events)

day_type = pd.DataFrame({
    "date": dates["date"],
    "day_type": dates["date"].apply(lambda x: "Holiday" if x.weekday() >= 5 else "Workday"),
    "locale": "Ordinary"
})

holidays = (
    pd.merge(day_type, holidays_fil, on="date", how="left", suffixes=("", "_new"))
    .assign(
        day_type=lambda df: df["day_type_new"].combine_first(df["day_type"]),
        locale=lambda df: df["locale_new"].combine_first(df["locale"])
    )
    .drop(columns=["day_type_new", "locale_new"])
    .drop_duplicates(subset=["date"], keep="first")
)


# Merging all dfs
comb_all = (
    pd.merge(train_test, stores, on="store_nbr", how="left")
    .merge(oil_price, on="date", how="left")
    .merge(transactions, on=["date", "store_nbr"], how="left")
    .merge(holidays, on="date", how="left")
    .set_index("id")
).rename(columns={"locale": "day_locale", "type": "store_type"})




# Transactions filtering
to_float64 = ["sales", "oil_price", "transactions", "onpromotion"]
to_category = ["family", "data_type", "store_nbr", "store_type", "cluster",
               "state", "city", "day_type", "day_locale", "weekday", "year_mo"]


def convert_types(df):
    df[to_float64] = df[to_float64].astype("float64")
    df[to_category] = df[to_category].astype("category")
    return df

desired_order = [
    "date", "sales", "data_type", "oil_price", "transactions",
    "onpromotion", "family", "store_nbr", "store_type", "cluster",
    "state", "city", "day_type", "day_locale", "weekday", "year_mo"
]

full_df = (
    convert_types(comb_all)
    .loc[lambda df: df["date"] >= "2013-01-16"]
    .reindex(columns=desired_order)
)

cat_df = full_df.copy()
full_df.loc[(full_df["sales"] == 0) & (full_df["transactions"].isna()), "transactions"] = 0
full_df[to_float64] = full_df[to_float64].apply(np.log1p)

label_encoder = LabelEncoder()
for col in to_category:
    full_df[col] = label_encoder.fit_transform(full_df[col])




# RF-transactions model
train_tr, test_tr = (
    full_df
    .pipe(lambda df: (
        df[df["transactions"].notna()],
        df[df["transactions"].isna()]
    ))
)

train_tr = train_tr.drop(columns=["date", "sales", "data_type"])
test_tr = test_tr.drop(columns=["date", "sales", "data_type"])

X = train_tr.drop(columns=["transactions"])
y = train_tr["transactions"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Val predictions - transactions
rf_model = RandomForestRegressor(n_estimators=200, random_state=42,  n_jobs=16, verbose=2)
rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_val)
rf_rmse = np.sqrt(mean_squared_error(y_val, rf_predictions))
print(f"Random Forest RMSE: {rf_rmse:.4f}")

cv_scores = cross_val_score(rf_model, X_train, y_train, cv=10, scoring="neg_mean_squared_error")
cv_rmse = np.sqrt(-cv_scores.mean())
print(f"Cross-Validation RMSE: {cv_rmse:.4f}")


# Test predictions - transactions
test_predictions = rf_model.predict(test_tr.drop(columns=["transactions"]))

trans_predictions = pd.DataFrame({
    "transactions": test_predictions
}, index=test_tr.index)




# Sales filtering
full_df.loc[trans_predictions.index, "transactions"] = trans_predictions["transactions"]
full_df["transactions_ma"] = full_df["transactions"].rolling(window=7, min_periods=1).mean()

columns_to_scale = [col for col in to_float64 if col != "sales"]
scaler = RobustScaler()
full_df[columns_to_scale] = scaler.fit_transform(full_df[columns_to_scale])
df_ff = full_df.copy()
df_ff[to_category] = cat_df[to_category]


# Fourier Series
def add_fourier_features(df, period=365, n_harmonics=5):
    df = df.copy()
    for i in range(1, n_harmonics + 1):
        df[f'sin_{i}'] = np.sin(2 * np.pi * i * df.index.dayofyear / period)
        df[f'cos_{i}'] = np.cos(2 * np.pi * i * df.index.dayofyear / period)
    return df

full_df["date"] = pd.to_datetime(full_df["date"])
full_df.set_index("date", inplace=True)
df_ff = add_fourier_features(full_df).reset_index(drop=True)




# Catboost-sales model
to_category = ["family", "store_nbr", "store_type", "cluster", "state", "city", "weekday", "year_mo"]

train_sales = df_ff[df_ff["data_type"] == "train"].drop(columns=["data_type", "date", "day_locale", "day_type"])
test_sales = df_ff[df_ff["data_type"] == "test"].drop(columns=["data_type", "date", "day_locale", "day_type"])

X = train_sales.drop(columns=["sales"])
y = train_sales["sales"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Optuna
def objective(trial):
    param = {
        'iterations': trial.suggest_int('iterations', 500, 2500),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'depth': trial.suggest_int('depth', 2, 6),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 0.0, 1.0),
        'border_count': trial.suggest_int('border_count', 1, 255)
    }

    model = CatBoostRegressor(**param, early_stopping_rounds=50, eval_metric="RMSE", task_type="GPU",
                               cat_features=to_category, verbose=2)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=2)

    preds = model.predict(X_val)
    rmse = root_mean_squared_error(y_val, preds)
    return rmse


study = optuna.create_study(sampler=optuna.samplers.CmaEsSampler(), direction="minimize")
study.optimize(objective, n_trials=50, show_progress_bar=True)

best_params = study.best_params


# Manual
# best_params = {
#     'iterations': 2188,
#     'learning_rate': 0.06955405529991802,
#     'depth': 6,
#     'l2_leaf_reg': 3.615285353491527,
#     'bagging_temperature': 0.3936185066584687,
#     'random_strength': 0.7759750334832344,
#     'border_count': 52
# }


# Val predictions - sales
best_model = CatBoostRegressor(**best_params, cat_features=to_category, eval_metric="RMSE", task_type="GPU", verbose=2)
best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
y_pred = best_model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
rmse = mse**0.5
print(f"Validation RMSE: {rmse:.4f}")


# Test predictions - sales
X_test = test_sales.drop(columns=["sales"])
test_predictions = best_model.predict(X_test)
test_predictions_exp = np.exp(test_predictions)

result = pd.DataFrame({
    "sales": test_predictions_exp
}, index=test_sales.index)

result.reset_index(inplace=True)


# Export result
result.to_csv("Store Sales/sales_result/sales_catboost.csv", index=False)