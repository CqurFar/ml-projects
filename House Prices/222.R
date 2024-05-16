# Hadley Wickham
library(conflicted)
library(ggplot2)
library(ggridges)
library(styler)
library(readxl)
library(dplyr)
library(janitor)
library(hexbin)
library(tidymodels)
library(mice) # mice
library(DMwR2) # KNN
library(DescTools) # RobScale

library(tidyverse)
library(scales)
library(ggrepel)
library(patchwork)
library(data.table)
library(Matrix)
library(glmnet)


# Maarten Speekenbrink
library(sdamr)
library(effectsize)
library(car)
library(GGally)
library(MASS)
library(mediation)
library(emmeans)
library(forcats)
library(tidyr)
library(afex)
library(lme4)

library(lavaan)
library(semPlot)


# Облака
library(tm)
library(wordcloud)
library(RColorBrewer)


# Моделирование
library(caret)
library(e1071)
library(glmnet)
library(gbm)
library(xgboost)
library(randomForest)
library(recommenderlab)

library(rpart)
library(rpart.plot)
library(mlr3)
library(ranger)


# Многопоточность
library(doParallel)
library(parallel)
library(foreach)


# Техническое
num_cores = detectCores()
registerDoParallel(cores = num_cores)
windowsFonts(TNR = windowsFont("Times New Roman"))
# sapply(data, function(y) sum(length(which(is.na(y)))))

conflicted::conflicts_prefer(dplyr::select)
conflicted::conflicts_prefer(dplyr::filter)
conflicted::conflicts_prefer(base::intersect)
conflicted::conflicts_prefer(base::as.matrix)
conflicted::conflicts_prefer(base::setdiff)
conflicted::conflicts_prefer(recipes::update)
conflicted::conflicts_prefer(stats::step)




# 01. Подготовка даных ----------------------------------------------------


# Импорт
data_train = read_csv("data/train.csv", na = c("N/A", "NA", "", ".")) |> 
  clean_names() |>
  select(id, sale_price, everything())

data_test = read_csv("data/test.csv", na = c("N/A", "NA", "", ".")) |> 
  clean_names() |>
  select(id, everything())


# Проверка соответствия колонок
{
# # Названия 
# cols_train = colnames(data_train)
# cols_test = colnames(data_test)
# 
# # Колонки, которые есть в train
# diff_df1_df2 = setdiff(cols_train, cols_test)
# print(paste("Колонки, которые есть в train, но отсутствуют в test:", paste(diff_df1_df2, collapse = ", ")))
# 
# # Колонки, которые есть в test
# diff_df2_df1 = setdiff(cols_test, cols_train)
# print(paste("Колонки, которые есть в test, но отсутствуют в train:", paste(diff_df2_df1, collapse = ", ")))
}

  # Проверка бинарных
{
# unique_counts = data_train |>
#   summarise_all(n_distinct)
# 
# binary_vars = names(unique_counts)[unique_counts == 2]
}


# Фильтрация
preprocess_data = function(data) {
  
  binary_cols = c("street", "utilities", "central_air", "x2nd_flr_sf", "low_qual_fin_sf", 
                  "half_bath", "fireplaces", "wood_deck_sf", "enclosed_porch", "x3ssn_porch", 
                  "screen_porch", "pool_area", "misc_val")
  
  binary_df = data[, binary_cols]
  
  
  # >= 50% нулей в бинарные
  threshold = 0.5
  convert_to_binary = function(df, threshold) {
    df |>
      mutate(across(where(is.numeric), ~ {
        if (mean(. == 0, na.rm = TRUE) >= threshold) {
          as.numeric(. != 0)
        } else {
          .
        }
      }))
  }
  
  binary = convert_to_binary(binary_df, threshold) |>
    mutate(street = if_else(street == "Grvl", 1, 0),
           utilities = if_else(is.na(utilities), 1, if_else(utilities == "AllPub", 1, 0)),
           central_air = if_else(central_air == "Y", 1, 0),
           id = data$id) |>
    select(id, everything())
  
  binary_inter = binary |>
    select(-street, -utilities, -central_air)
  
  
  # ДФ по типам
  numeric = data |>
    select_if(is.numeric) |>
    select(-intersect(names(data), names(binary_inter)), id) |>
    select(-ms_sub_class, -overall_qual, -overall_cond) |>
    knnImputation(k=50) |>
    mutate(across(where(is.numeric), round)) |>
    select(id, everything())
  
  factor = data |>
    select(-intersect(names(data), names(numeric)), id) |>
    select(-intersect(names(data), names(binary)), id) |>
    select(id, everything()) |>
    mutate_all(~ as.factor(.)) |>
    mutate(id = as.numeric(as.character(id)))
  
  
  # Remove cols >20% na, ост - мода
  na_percent = sapply(factor, function(x) sum(is.na(x))/length(x))
  columns_to_remove = names(na_percent)[na_percent > 0.2]
  
  get_mode = function(v) {
    uniqv = unique(v)
    uniqv[which.max(tabulate(match(v, uniqv)))]
  }
  
  factor = factor |>
    select(-one_of(columns_to_remove)) |>
    mutate_if(is.factor, function(x) replace_na(x, as.character(get_mode(x))))
  
  
  # Замена всех редких уровней на Other
  factor_col = colnames(factor)
  for (col in factor_col) {
    
    freq = table(factor[[col]]) / nrow(factor)
    rare_levels = names(freq)[freq < 0.05]
    
    levels(factor[[col]])[levels(factor[[col]]) %in% rare_levels] = "Other"
  }
  
  
  # Сборка
  all_data = Reduce(function(x, y) merge(x, y, by = "id", all.x = TRUE), 
                    list(numeric, binary, factor))
  
  return(all_data)
}


train = preprocess_data(data_train)

test = preprocess_data(data_test) |>
  select(id, everything())


# Перебираем каждый фактор и проверяем различия в уровнях между train и test
factor_names_test = names(Filter(is.factor, test))

for (factor_name in factor_names_test) {
  train_levels = levels(train[[factor_name]])
  test_levels = levels(test[[factor_name]])
  
  diff_levels = setdiff(train_levels, test_levels)
  
  if (length(diff_levels) > 0) {
    cat("Фактор", factor_name, "имеет новые уровни:", diff_levels, "\n")
  }
}




# 02. Переменные  с высокой кореляцией ------------------------------------


# Числовые
correlation_threshold = 0.3
correlation_matrix = cor(train[sapply(train, is.numeric)])
numeric_columns_to_keep = names(which(abs(correlation_matrix[, "sale_price"]) > correlation_threshold))
numeric_columns_to_keep = setdiff(numeric_columns_to_keep, "sale_price")


# Факторные
factor_vars = names(train)[sapply(train, is.factor)]
significant_factor_vars = character(0)

for (var in factor_vars) {
  fit = aov(as.formula(paste("sale_price ~", var)), data = train)
  p_value = summary(fit)[[1]]$"Pr(>F)"[1]
  if (p_value < 0.05) {
    significant_factor_vars = c(significant_factor_vars, var)
  }
}


# Значимые переменные
final_columns = c("sale_price", numeric_columns_to_keep, significant_factor_vars)
final_train = train[, final_columns]

col = intersect(names(final_train), names(test))
final_test = test[, col] |>
  mutate(sale_price = 0)


# Объединение 
combined_data = rbind(final_train, final_test)

factor_vars = names(combined_data)[sapply(combined_data, is.factor)]
for (var in factor_vars) {
  combined_data[[var]] = factor(combined_data[[var]])
}


# Разделение
numeric_vars = train |>
  select(where(is.numeric)) |>
  select(-sale_price)

means = sapply(numeric_vars, mean, na.rm = TRUE)
sds = sapply(numeric_vars, sd, na.rm = TRUE)


# дф
train_new = combined_data[1:nrow(final_train), ] |>
  select(-garage_yr_blt)

train_new$sale_price = log(train_new$sale_price + 1)

test_new = combined_data[(nrow(final_train) + 1):nrow(combined_data), ] |>
  select(-sale_price, -garage_yr_blt)




# 02. LM ------------------------------------------------------------------


# Модель
mod = lm(sale_price ~ ., data = train_new)
summary(mod)


# Выбросы
outliers = outlierTest(mod)
outlier_indices = as.numeric(names(outliers$rstudent))

Q1 = quantile(train_new$sale_price, 0.25)
Q3 = quantile(train_new$sale_price, 0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outlier_indices = which(train_new$sale_price < lower_bound | train_new$sale_price > upper_bound)


# Очищенная модель 
train_clean = train_new |>
  filter(!(row_number() %in% outlier_indices))

train_clean = train_clean[-outlier_indices, ]

mod_cl = lm(sale_price ~ ., data = train_clean)
summary(mod_cl)


# Lasso
x = model.matrix(sale_price ~ ., data = train_clean)[,-1]
y = train_clean$sale_price

lasso_mod = cv.glmnet(x, y, alpha = 1, nfolds = 50)
predicted = predict(lasso_mod, s = lasso_mod$lambda.min, newx = model.matrix(~ ., data = test_new)[,-1]) |>
  round(-2)

predicted = as.vector(predicted)


# Экспорт
result_LM = test |>
  add_column(predicted_sale_price = predicted, .after = 1) |>
  select(id, predicted_sale_price)

kaggle_lm = result_LM |>
  rename(Id = id, SalePrice = predicted_sale_price)

View(kaggle_lm)
# write.csv(kaggle_lm, "kaggle_lm.csv", row.names = FALSE)




# RF ----------------------------------------------------------------------


# Первичная модель
rf_model = randomForest(sale_price ~ ., data = train_clean, importance = TRUE)
importance = randomForest::importance(rf_model) |>
  as.data.frame()


# Важные переменные
feature_importance = data.frame(Feature = rownames(importance), Importance = importance[, '%IncMSE'])

top_features = feature_importance |>
  arrange(desc(Importance))
print(top_features)

important_threshold = 10
selected_features = top_features$Feature[top_features$Importance > important_threshold]


train_selected = train_clean[, c("sale_price", selected_features)]
test_selected = test_new[, c(selected_features)]


# Кросс-валидация
control_rf = trainControl(method = "cv", number = 50)
rf_cv = train(sale_price ~ ., data = train_selected, method = "rf", trControl = control_rf, allowParallel=TRUE)


# Новая модель
rf_model_selected = randomForest(sale_price ~ ., data = train_selected, 
                                 ntree = 1000, mtry = 25, min_n = 4, allowParallel = TRUE)

rf_predicted = predict(rf_model_selected, newdata = test_selected) |>
  unlist() |>
  exp() - 1

rf_predicted = as.vector(rf_predicted)  |>
  round(-2)


# Экспорт
result_rf = test |>
  add_column(predicted_sale_price = rf_predicted, .after = 1) |>
  select(id, predicted_sale_price)

kaggle_rf = result_rf |>
  rename(Id = id, SalePrice = predicted_sale_price)

View(kaggle_rf)
# write.csv(kaggle_rf, "kaggle_rf.csv", row.names = FALSE)




# GB ----------------------------------------------------------------------


# Спеки
tune_test = expand.grid(n.trees = c(100, 200, 300, 400, 500, 1000),
                        interaction.depth = c(1, 3, 5, 7, 10),
                        shrinkage = c(0.01, 0.1, 0.2),
                        n.minobsinnode = c(5, 10, 15, 20, 25, 30))

tune_grid = expand.grid(n.trees = c(1000),
                        interaction.depth = c(10),
                        shrinkage = c(0.01),
                        n.minobsinnode = c(20)) 

control = trainControl(method = "cv", number = 30)
gbm_fit = train(sale_price ~ ., data = train_selected, method = "gbm", 
                 trControl = control, tuneGrid = tune_grid, verbose = FALSE)


# Предсказание
gbm_predicted = predict(gbm_fit, test_selected, type = "raw") |>
  unlist() |>
  exp() - 1

gbm_predicted = as.vector(gbm_predicted)  |>
  round(-2)


# Экспорт
result_gb = test |>
  add_column(predicted_sale_price = gbm_predicted, .after = 1) |>
  select(id, predicted_sale_price)

kaggle_gb = result_gb |>
  rename(Id = id, SalePrice = predicted_sale_price)

View(kaggle_gb)
# write.csv(kaggle_gb, "kaggle_gb.csv", row.names = FALSE)