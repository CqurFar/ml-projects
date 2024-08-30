# Библиотеки
import numpy as np
import pandas as pd
import xgboost as xgb
import streamlit as st
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.stats import shapiro, boxcox
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix




# Streamlit Bar
st.set_page_config(layout="wide")
st.sidebar.subheader("Parameters Bar")

max_depth = st.sidebar.slider("Max Depth", min_value=3, max_value=10, value=5)
subsample = st.sidebar.slider("Subsample", min_value=0.6, max_value=1.0, value=0.8, step=0.1)
n_estimators = st.sidebar.slider("Number of Estimators", min_value=100, max_value=1000, value=1000)
learning_rate = st.sidebar.slider("Learning Rate", min_value=0.01, max_value=0.3, value=0.1, step=0.01)
colsample_bytree = st.sidebar.slider("Colsample by Tree", min_value=0.6, max_value=1.0, value=0.8, step=0.1)

apply_smote = st.sidebar.checkbox("Apply SMOTE Balancing", value=True)
apply_boxcox = st.sidebar.checkbox("Apply Box-Cox Standardization", value=True)
apply_feature_weights = st.sidebar.checkbox("Apply Feature Weights", value=False)




# Импорт
file_path = r"C:\Users\Art\Desktop\Dz\ВУЗ\РАБОТА\Python\Aurelien Geron\Wine\wine_data\WineQT.csv"
data = pd.read_csv(file_path, na_values=["N/A", "NA", "NaN", "", "."]).set_index("Id")


# Убираем редкие уровни
min_class_count = 22
class_counts = data["quality"].value_counts()
valid_classes = class_counts[class_counts >= min_class_count].index

data = data[data["quality"].isin(valid_classes)]


# Перекодировка уровней
label_encoder = LabelEncoder()
data["quality"] = label_encoder.fit_transform(data["quality"])


# Кодирование переменных
to_float64 = [col for col in data.columns if col != "quality"]
to_category = ["quality"]


def convert_types(df, float_cols, cat_cols):
    df[float_cols] = df[float_cols].astype("float64")
    df[cat_cols] = df[cat_cols].astype("category")
    return df

data = convert_types(data, to_float64, to_category)

columns = ["quality"] + [col for col in data.columns if col != "quality"]
data = data[columns]


# Проверка нормальности распределения
def check_normality(df, numeric_cols):
    normality_results = {}
    for col in numeric_cols:
        shapiro_test = shapiro(df[col])
        normality_results[col] = shapiro_test.pvalue
    return normality_results

numeric_cols = data.select_dtypes(include=["float64"]).columns
normality_results = check_normality(data, numeric_cols)


# Анализ результатов теста Шапиро-Уилка
all_normal = True
non_normal_columns = []
for col, p_value in normality_results.items():
    if p_value < 0.05:
        all_normal = False
        non_normal_columns.append(col)

if all_normal:
    normality_message = "Переменные распределены нормально (не отвергаем H0)"
else:
    if non_normal_columns:
        normality_message = f"Переменные {', '.join(non_normal_columns)} не распределены нормально"
    else:
        normality_message = "Все переменные не распределены нормально (отвергаем H0)"


# Выявление выбросов
def detect_outliers(df, numeric_cols):
    outliers = {}
    for col in numeric_cols:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        outliers[col] = np.where(z_scores > 3)[0]
    return outliers


# Замена выбросов медианой по переменной quality
def replace_outliers_with_median(df, numeric_cols, quality_col):
    df_cleaned = df.copy()
    for col in numeric_cols:
        for quality in df[quality_col].cat.categories:
            subset = df[df[quality_col] == quality]
            median = subset[col].median()
            z_scores = np.abs((subset[col] - subset[col].mean()) / subset[col].std())
            outliers = np.where(z_scores > 3)[0]
            df_cleaned.loc[(df_cleaned[quality_col] == quality).values & (np.isin(df.index, subset.index[outliers])), col] = median
    return df_cleaned

data = replace_outliers_with_median(data, numeric_cols, "quality")


# Стандартизация Box-Cox
def boxcox_transform(df, numeric_cols):
    df_transformed = df.copy()
    for col in numeric_cols:
        df_transformed[col], _ = boxcox(df_transformed[col] + 1)
    return df_transformed

if apply_boxcox:
    data = boxcox_transform(data, numeric_cols)




# Разделение на наборы
X = data.drop("quality", axis=1)
y = data["quality"].astype("category").cat.codes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Балансировка SMOTE
if apply_smote:
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)


# Модель XGBoost
num_classes = len(np.unique(y_train))

xgb_clf = xgb.XGBClassifier(
    learning_rate=learning_rate,
    max_depth=max_depth,
    n_estimators=n_estimators,
    subsample=subsample,
    colsample_bytree=colsample_bytree,
    objective="multi:softprob",
    num_class=num_classes,
    random_state=42
)

xgb_clf.fit(X_train, y_train)
y_pred_xgb = xgb_clf.predict(X_test)
y_pred_proba_xgb = xgb_clf.predict_proba(X_test)


# Изменение весов
if apply_feature_weights:
    importance = xgb_clf.feature_importances_
    feature_weights = np.ones_like(importance)
    top_indices = np.argsort(importance)[-3:]
    feature_weights[top_indices] = 1000
    X_train = X_train * feature_weights
    X_test = X_test * feature_weights




# Модель метрик
def evaluate_model(y_test, y_pred, y_pred_proba, model_name):
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    report = classification_report(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)

    return accuracy, roc_auc, report, conf_mat

accuracy_xgb, roc_auc_xgb, report_xgb, conf_mat_xgb = evaluate_model(y_test, y_pred_xgb, y_pred_proba_xgb, "XGBoost")


# Вывод
st.write("# Wine Quality")

st.write("## XGBoost")
st.write(f"**Accuracy:** {accuracy_xgb:.4f}")
st.write(f"**ROC AUC:** {roc_auc_xgb:.4f}")
st.write("### Classification Report")
st.text(report_xgb)


col1, col2 = st.columns(2)

# Матрица ошибок
with col1:
    st.write("## Confusion Matrix")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_mat_xgb, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

# Переменные
with col2:
    st.write("## Feature Importance")
    importance = xgb_clf.feature_importances_
    features = X.columns
    importance_df = pd.DataFrame({"Feature": features, "Importance": importance})
    fig = px.bar(importance_df.sort_values(by="Importance", ascending=False), x="Importance", y="Feature", title="Feature Importance")
    fig.update_layout(height=600)
    st.plotly_chart(fig)