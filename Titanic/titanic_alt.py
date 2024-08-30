import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV




# Импорт
data_train = pd.read_csv("Titanic/titanic_data/titanic_train.csv", na_values=["N/A", "NA", "NaN", "", "."]).set_index("PassengerId")
data_test = pd.read_csv("Titanic/titanic_data/titanic_test.csv", na_values=["N/A", "NA", "NaN", "", "."]).set_index("PassengerId")


# Убираем ненужные переменные
to_drop = ["Name", "Ticket", "Cabin"]

data_train = data_train.drop(columns=to_drop)
data_test = data_test.drop(columns=to_drop)


# Замена NA
def na_replace(df):

    mean_age_men = df[df["Sex"] == "male"]["Age"].mean()
    mean_age_women = df[df["Sex"] == "female"]["Age"].mean()

    df.loc[(df["Sex"] == "male") & (df["Age"].isnull()), "Age"] = mean_age_men
    df.loc[(df["Sex"] == "female") & (df["Age"].isnull()), "Age"] = mean_age_women

    mode_embarked = df["Embarked"].mode()[0]
    df["Embarked"] = df["Embarked"].fillna(mode_embarked)

    return df


data_train = na_replace(data_train)
data_test = na_replace(data_test)


# Новые переменные
data_train["AgeBucket"] = data_train["Age"] // 15 * 15
data_test["AgeBucket"] = data_test["Age"] // 15 * 15

data_train["RelativesOnboard"] = data_train["SibSp"] + data_train["Parch"]
data_test["RelativesOnboard"] = data_test["SibSp"] + data_test["Parch"]


# Кодирование переменных
to_float64 = ["Fare"]
to_int64 = ["Age", "SibSp", "Parch"]
to_category = ["AgeBucket", "Pclass", "Sex", "Embarked"]


def convert_types(df):
    df[to_float64] = df[to_float64].astype('float64')
    df[to_int64] = df[to_int64].astype('int64')
    df[to_category] = df[to_category].astype('category')
    return df


data_train = convert_types(data_train)
data_test = convert_types(data_test)


# Кодирование и мастабирование
data_train = pd.get_dummies(data_train, columns=["Pclass", "Sex", "Embarked"], drop_first=False, dtype="int64")
data_test = pd.get_dummies(data_test, columns=["Pclass", "Sex", "Embarked"], drop_first=False, dtype="int64")

scaler = StandardScaler()
data_train[to_float64 + to_int64] = scaler.fit_transform(data_train[to_float64 + to_int64])
data_test[to_float64 + to_int64] = scaler.transform(data_test[to_float64 + to_int64])


# Убираем ненужные переменные
to_drop = ["Age", "SibSp", "Parch"]

data_train = data_train.drop(columns=to_drop)
data_test = data_test.drop(columns=to_drop)




# Модель
X_train = data_train.drop("Survived", axis=1)
y_train = data_train["Survived"]

forest_reg = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_features': [None, 'auto', 'sqrt', 'log2'],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}
grid_search = GridSearchCV(forest_reg, param_grid, cv=10, scoring="accuracy",
                           return_train_score=True, n_jobs=6, verbose=2)
grid_search.fit(X_train, y_train)

print("Лучшие параметры:", grid_search.best_params_)
print("Лучшая точность:", grid_search.best_score_)


# Предсказания
X_test = data_test
y_test = grid_search.best_estimator_.predict(X_test)

results = pd.DataFrame({
    "PassengerId": data_test.index,
    "Survived": y_test
})


# Экспорт
results.to_csv("titanic_rf.csv", index=False)