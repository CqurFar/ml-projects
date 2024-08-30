# Libraries
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import load
from janitor import clean_names
from pmdarima import auto_arima
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler




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
    .reindex(columns=desired_order)
)

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


# Import RF model and test predictions
rf_model_load = load("Store Sales/sales_models/rf_trans.joblib")

test_predictions = rf_model_load.predict(test_tr.drop(columns=["transactions"]))
trans_predictions = pd.DataFrame({
    "transactions": test_predictions
}, index=test_tr.index)




# Sales filtering
full_df.loc[trans_predictions.index, "transactions"] = trans_predictions["transactions"]
full_df["transactions_ma"] = full_df["transactions"].rolling(window=7, min_periods=1).mean()

columns_to_scale = [col for col in to_float64 if col != "sales"]
scaler = RobustScaler()
full_df[columns_to_scale] = scaler.fit_transform(full_df[columns_to_scale])
df_sales = full_df.copy()


# Weekly and two-week lag
df_sales.reset_index(inplace=True)
df_sales["lag_7"] = df_sales.groupby(["family", "store_nbr"], observed=False)["sales"].shift(7)
df_sales["lag_14"] = df_sales.groupby(["family", "store_nbr"], observed=False)["sales"].shift(14)

train_sales = df_sales[df_sales["data_type"] == 1].drop(columns=["data_type"])
test_sales = df_sales[df_sales["data_type"] == 0].drop(columns=["data_type"])
train_sales = train_sales.dropna(subset=["lag_7", "lag_14"])




# Arima-sales model
forecast_dfs = []
total_iterations = train_sales.groupby(["family", "store_nbr"]).ngroups

# Tqdm progress bar
with tqdm(total=total_iterations) as pbar:
    for (family, store_nbr), group in train_sales.groupby(["family", "store_nbr"]):
        group = group.set_index("date")
        group.reset_index(inplace=True)

        exog_train = group[["lag_7", "lag_14", "oil_price", "transactions", "transactions_ma", "onpromotion",
                            "day_type", "day_locale", "weekday", "year_mo"]]
        model = auto_arima(group["sales"], exogenous=exog_train, seasonal=False, m=365, stepwise=False,
                           trace=False, error_action="ignore",
                           suppress_warnings=True, n_jobs=16)

        test_group = test_sales[(test_sales["family"] == family) & (test_sales["store_nbr"] == store_nbr)]
        test_group = test_group.set_index("date")
        test_group.reset_index(inplace=True)

        # Test predictions with iteratively lag
        group_predictions = []
        for i in range(len(test_group)):
            exog_test = test_group[["lag_7", "lag_14", "oil_price", "transactions", "transactions_ma", "onpromotion",
                                    "day_type", "day_locale", "weekday", "year_mo"]].iloc[[i]]
            forecast = model.predict(n_periods=1, exogenous=exog_test)

            forecast_values = forecast.values
            group_predictions.append(forecast_values[0])

            if i < len(test_group) - 1:
                test_group.loc[test_group.index[i + 1], "lag_7"] = forecast_values[0]
                test_group.loc[test_group.index[i + 1], "lag_14"] = test_group["lag_7"].iloc[i]

        test_group["sales"] = group_predictions
        forecast_dfs.append(test_group)

        pbar.update(1)


# Merging into a common df
result = pd.concat(forecast_dfs).reset_index(drop=True)
result["sales"] = np.expm1(result["sales"])
result["sales"] = result["sales"].round(3)
result["sales"] = result["sales"].clip(lower=0)
result = result[["id", "sales"]]

# Export result
result.to_csv("Store Sales/sales_result/sales_arima.csv", index=False)