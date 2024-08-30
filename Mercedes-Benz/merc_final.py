import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
from janitor import clean_names
from scipy.sparse import dok_matrix
from sklearn.manifold import Isomap
from sklearn.model_selection import KFold
from category_encoders import JamesSteinEncoder
from tensorflow.keras import layers, optimizers, regularizers, Sequential




# Импорт данных
def import_csv(directory="data"):
    files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(".csv")]
    data_dict = {}
    for file in files:
        df = clean_names(pd.read_csv(file, na_values=["N/A", "NA", "", "."]))
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        data_name = os.path.splitext(os.path.basename(file))[0]
        data_dict[data_name] = df
    return data_dict


# Объединение train и test наборов данных
def merge_train_test(train, test):
    train["data_type"] = "train"
    test["data_type"] = "test"
    train["id"] = train["id"].astype(str) + "-train"
    test["id"] = test["id"].astype(str) + "-test"
    return pd.concat([train, test], ignore_index=True)


# Переименование и приведение типов столбцов
def rename_and_cast_columns(df):
    object_columns = df.select_dtypes(include=["object"]).columns.difference(["id", "data_type"])
    int_columns = df.select_dtypes(include=["integer", "int"]).columns
    new_column_names = {**{col: f"cat-{i+1:02d}" for i, col in enumerate(object_columns)},
                        **{col: f"bin-{i+1:02d}" for i, col in enumerate(int_columns)}}
    if "y" in df.columns:
        new_column_names["y"] = "target"
    df.rename(columns=new_column_names, inplace=True)

    for col in df.columns:
        if col.startswith("bin-"):
            df[col] = df[col].astype("int8")
        elif col.startswith("cat-"):
            df[col] = df[col].astype("category")

    column_order = ["id", "data_type"] + (["target"] if "target" in df.columns else []) + \
                   [col for col in df.columns if col not in ["id", "data_type", "target"]]
    df = df[column_order]
    return df


# Кодирование категориальных переменных
def js_encoding(df):
    cat_columns = [col for col in df.columns if col.startswith("cat-")]
    train_df = df[df["data_type"] == "train"].copy()
    test_df = df[df["data_type"] == "test"].copy()

    for col in cat_columns:
        train_categories = set(train_df[col].cat.categories)
        test_categories = set(test_df[col].cat.categories)
        unknown_categories = test_categories - train_categories

        if unknown_categories:
            mode_value = train_df[col].mode()[0]
            all_categories = train_categories.union(test_categories)
            train_df[col] = train_df[col].cat.set_categories(all_categories)
            test_df[col] = test_df[col].cat.set_categories(all_categories)
            test_df[col] = test_df[col].apply(lambda x: mode_value if x not in train_categories else x)

    encoder = JamesSteinEncoder(cols=cat_columns)
    train_df[cat_columns] = encoder.fit_transform(train_df[cat_columns], train_df["target"])
    test_df[cat_columns] = encoder.transform(test_df[cat_columns])

    train_df[cat_columns] = train_df[cat_columns].astype("float64")
    test_df[cat_columns] = test_df[cat_columns].astype("float64")

    df = pd.concat([train_df, test_df], ignore_index=True)
    return df


# Уменьшение размерности с использованием Isomap
def isomap(df, n_components: int, n_neighbors: int):
    binary_columns = [col for col in df.columns if col.startswith("bin-")]
    X = df[binary_columns].values
    X_sparse = dok_matrix(X)

    while True:
        try:
            isomap = Isomap(n_components=n_components, n_neighbors=n_neighbors)
            X_reduced = isomap.fit_transform(X_sparse)
            break
        except UserWarning:
            n_neighbors += 5

    df_reduced = pd.DataFrame(X_reduced, columns=[f"isomap-{i + 1}" for i in range(n_components)], index=df.index)
    df_combined = pd.concat([df.drop(columns=binary_columns), df_reduced], axis=1)
    return df_combined


# Подготовка данных для моделирования
def prepare_for_modeling(df):
    train_df = df[df["data_type"] == "train"].copy()
    test_df = df[df["data_type"] == "test"].copy()
    train_df.drop(columns=["id", "data_type"], inplace=True)
    test_df.drop(columns=["id", "data_type"], inplace=True)
    train_df["target"] = np.log1p(train_df["target"])

    y_train = train_df["target"]
    X_train = train_df.drop(columns=["target"])
    X_test = test_df.drop(columns=["target"])
    return X_train, y_train, X_test


# Создание модели
def build_model(hp, input_shape):
    activation = hp.Choice("activation", values=["relu", "swish"])
    n_hidden = hp.Int("n_hidden", min_value=2, max_value=12)
    n_neurons = hp.Int("n_neurons", min_value=16, max_value=512, step=16)
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=0.01, sampling="log")
    regularizer_l2 = hp.Float("regularizer_l2", min_value=0.0, max_value=0.2, sampling="linear")
    dropout_rate = hp.Float("dropout_rate", min_value=0.0, max_value=0.8, sampling="linear")

    optimizer_choice = hp.Choice("optimizer", values=["adam", "rmsprop"])
    if optimizer_choice == "adam":
        optimizer = optimizers.Adam(learning_rate=learning_rate)
    else:
        optimizer = optimizers.RMSprop(learning_rate=learning_rate)

    model = Sequential([
        layers.Input(shape=[input_shape]),
        layers.BatchNormalization(),
        *[layers.Dense(n_neurons, activation=activation,
                       kernel_initializer="he_normal",
                       kernel_regularizer=regularizers.l2(regularizer_l2))
          for _ in range(n_hidden)],
        layers.Dropout(rate=dropout_rate),
        layers.BatchNormalization(),
        layers.Dense(1)
    ])

    model.compile(optimizer=optimizer, loss="mse", metrics=["RootMeanSquaredError"])
    return model


# Обучение модели
def train_model(X_train, y_train, n_splits: int, n_epochs: int, batch_size: int):
    X_train_np = X_train.to_numpy()
    y_train_np = y_train.to_numpy()

    tuner = kt.BayesianOptimization(
        lambda hp: build_model(hp, X_train.shape[1]),
        max_trials=30,
        directory="model",
        project_name="bayesian_optimization",
        objective="val_RootMeanSquaredError",
        seed=42,
        overwrite=True
    )

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    val_scores = []

    for train_index, val_index in kfold.split(X_train_np):
        X_train_fold, X_val_fold = X_train_np[train_index], X_train_np[val_index]
        y_train_fold, y_val_fold = y_train_np[train_index], y_train_np[val_index]

        tuner.search(
            X_train_fold, y_train_fold,
            epochs=n_epochs,
            validation_data=(X_val_fold, y_val_fold),
            batch_size=batch_size,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
        )

        best_model = tuner.get_best_models(num_models=1)[0]

        val_loss, val_rmse = best_model.evaluate(X_val_fold, y_val_fold, verbose=0)
        print(f"RMSE на валидационном фолде: {val_rmse:.4f}")

        val_scores.append(val_rmse)

    avg_val_rmse = np.mean(val_scores)
    print(f"Средний RMSE по всем фолдам: {avg_val_rmse:.4f}")

    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.fit(X_train_np, y_train_np, epochs=n_epochs, batch_size=batch_size, verbose=0)

    best_params = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Лучшие гиперпараметры:", best_params.values)
    return best_model, best_params, avg_val_rmse


# Применение Монте-Карло Dropout
def mc_dropout(model, X_test, n_samples=100):
    y_probas = np.stack([model(X_test, training=True) for _ in range(n_samples)])
    y_proba = y_probas.mean(axis=0)
    uncertainty = y_probas.std(axis=0)
    return y_proba, uncertainty


# Экстракция тестового набора данных
def extract_test(df):
    test_df = df[df["data_type"] == "test"].copy()
    return test_df


# Создание файла для отправки
def submission(predictions, test_df, output_directory="submission", output_file="submission.csv"):
    predictions = np.expm1(predictions)
    predictions = np.round(predictions, 2)
    submission_df = test_df[["id"]].copy()
    submission_df["y"] = predictions
    submission_df["id"] = submission_df["id"].str.extract(r"(\d+)")
    output_path = f"{output_directory}/{output_file}"
    submission_df.to_csv(output_path, index=False)
    print(f"Submission сохранён в папку {output_path}")


# Основная функция обработки и моделирования
def processing(directory="data"):
    data_dict = import_csv(directory)
    train = data_dict["train"]
    test = data_dict["test"]

    df = merge_train_test(train, test)
    df = rename_and_cast_columns(df)
    df = js_encoding(df)
    df = isomap(df, n_components=46, n_neighbors=10)

    X_train, y_train, X_test = prepare_for_modeling(df)
    best_model, best_params, avg_val_rmse = train_model(X_train, y_train, n_splits=12, n_epochs=100, batch_size=256)
    predictions, uncertainty = mc_dropout(best_model, X_test, n_samples=100)
    test_df = extract_test(df)
    submission(
        predictions, test_df,
        output_directory="result",
        output_file="merc_dnn_v5.csv"
    )


# Запуск
processed = processing("data")
