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
# na_count = sapply(data, function(y) sum(length(which(is.na(y)))))

conflicted::conflicts_prefer(dplyr::select)
conflicted::conflicts_prefer(dplyr::filter)
conflicted::conflicts_prefer(base::intersect)
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


# Проверка бинарных
unique_counts = data_train |>
  summarise_all(n_distinct)

binary_vars = names(unique_counts)[unique_counts == 2]


# Фильтрация
preprocess_data = function(data) {
  # Дф по типам + замена na
  numeric = data |>
    select_if(is.numeric) |>
    select(-ms_sub_class, -overall_qual, -overall_cond) |>
    knnImputation(k=50) |>
    mutate(across(where(is.numeric), round))
  
  binary = data |>
    select(id, street, utilities, central_air) |>
    mutate(street = if_else(street == "Grvl", 1, 0),
           utilities = if_else(utilities == "AllPub", 1, 0),
           central_air = if_else(central_air == "Y", 1, 0))
  
  factor = data |>
    select(-intersect(names(data), names(numeric)), id) |>
    select(id, everything(), -street, -utilities, -central_air) |>
    mutate_all(~ as.factor(.)) |>
    mutate(id = as.numeric(as.character(id)))
  
  
  # Remove cols >30% na, ост - мода
  na_percent = sapply(factor, function(x) sum(is.na(x))/length(x))
  columns_to_remove = names(na_percent)[na_percent > 0.3]
  
  get_mode = function(v) {
    uniqv = unique(v)
    uniqv[which.max(tabulate(match(v, uniqv)))]
  }
  
  factor = factor |>
    select(-one_of(columns_to_remove)) |>
    mutate_if(is.factor, function(x) replace_na(x, as.character(get_mode(x))))
  
  
  # Сборка
  all_data = Reduce(function(x, y) merge(x, y, by = "id", all.x = TRUE), 
                    list(numeric, binary, factor))
  return(all_data)
}

train = preprocess_data(data_train)
test = preprocess_data(data_test) |>
  select(-roof_matl, -heating)




# 02. New Levels ----------------------------------------------------------


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

columns_to_process = c("ms_sub_class", "ms_zoning", "lot_shape", "land_contour", "lot_config", 
                       "land_slope", "neighborhood", "condition1", "condition2", "bldg_type", 
                       "house_style", "overall_qual", "overall_cond", "roof_style", "roof_matl", 
                       "exterior1st", "exterior2nd", "mas_vnr_type", "exter_qual", "exter_cond", 
                       "foundation", "bsmt_qual", "bsmt_cond", "bsmt_exposure", "bsmt_fin_type1", 
                       "bsmt_fin_type2", "heating", "heating_qc", "electrical", "kitchen_qual", 
                       "functional", "garage_type", "garage_finish", "garage_qual", "garage_cond", 
                       "paved_drive", "sale_type", "sale_condition")

# Замена всех редких уровней на Other
for (col in columns_to_process) {

  freq = table(train[[col]]) / nrow(train)
  rare_levels = names(freq)[freq < 0.05]
  
  levels(train[[col]])[levels(train[[col]]) %in% rare_levels] = 'Other'
  levels(test[[col]])[!(levels(test[[col]]) %in% levels(train[[col]]))] = 'Other'
}




# 03. LM ------------------------------------------------------------------


# Модель
mod = lm(sale_price ~ ., data = train)
summary(mod)
confint(mod)

errors = residuals(mod)
predictg = predict(mod)


# Графки ошибок
hist(errors)
plot(mod)
plot(predictg, errors, xlab = "predicted", ylab = "residual")
abline(h=0,lty=3)


# # Искл стат не знач перемееные
# p_values = summary(mod)$coefficients[, "Pr(>|t|)"]
# variables_to_remove = names(p_values)[p_values > 0.05]

# Искл выбросы
outliers = outlierTest(mod)
influencePlot(mod, method = "indentity")
outlier_indices = as.numeric(names(outliers$rstudent))


# Модель очищенная
train_clean = train |>
  filter(!(row_number() %in% outlier_indices))

mod_cl = lm(sale_price ~ ., data = train_clean)
summary(mod_cl)


# Проверка
predicted_prices_train = predict(mod_cl, newdata = train_clean)
actual_prices_train = train_clean$sale_price

# Среднеквадратичная ошибка (RMSE)
rmse = sqrt(mean((actual_prices_train - predicted_prices_train)^2))
cat("RMSE for train:", rmse, "\n")

# Средняя абсолютная ошибка (MAE)
mae = mean(abs(actual_prices_train - predicted_prices_train))
cat("MAE for train:", mae, "\n")

# Коэффициент детерминации (R²)
r2 = cor(actual_prices_train, predicted_prices_train)^2
cat("R² for train:", r2, "\n")


# Готовое
predicted = predict(mod_cl, newdata = test) |>
  round(-2)

result_v1_LM = test |>
  add_column(predicted_sale_price = predicted, .after = 1) |>
  select(id, predicted_sale_price)

kaggle_lm = result_v1_LM |>
  rename(Id = id, SalePrice = predicted_sale_price)

View(kaggle_lm)
# write.csv(kaggle_lm, "kaggle_lm.csv", row.names = FALSE)




# RF ----------------------------------------------------------------------


# Кросс-валидация
control_rf = trainControl(method = "cv", number = 50)
rf_cv = train(sale_price ~ ., data = train_clean, method = "rf", trControl = control_rf, allowParallel=TRUE)


# Случайный лес
rf_model = randomForest(sale_price ~ ., data = train_clean, 
                        ntree = 1000, mtry = 58, allowParallel=TRUE)
rf_predicted = predict(rf_model, newdata = test) |>
  round(-2)


# Экспорт
result_v2_RF = test |>
  add_column(predicted_sale_price = rf_predicted, .after = 1) |>
  select(id, predicted_sale_price)

kaggle_rf = result_v2_RF |>
  rename(Id = id, SalePrice = predicted_sale_price)

View(kaggle_rf)
# write.csv(kaggle_rf, "kaggle_rf.csv", row.names = FALSE)




# GB ----------------------------------------------------------------------


# Кросс-валидация
control_xgb = trainControl(method = "cv", number = 50)
xgb_cv = train(sale_price ~ ., data = train_clean, method = "xgbTree", trControl = control_xgb, allowParallel=TRUE)


# Модель
xgb_data = xgb.DMatrix(data = model.matrix(sale_price ~ ., train_clean)[,-1], label = train_clean$sale_price)
xgb_params = list(objective = "reg:squarederror", max_depth = 2, eta = 0.4, nthread = 2,
                   gamma = 0, colsample_bytree =0.8, min_child_weight = 1, subsample = 1)

xgb_model = xgb.train(params = xgb_params, data = xgb_data, nrounds = 1000)
xgb_predicted = predict(xgb_model, newdata = xgb.DMatrix(model.matrix(~ . - 1, test))) |>
  round(-2)