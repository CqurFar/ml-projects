# Библиотеки
{
# Hadley Wickham
library(ggplot2)
library(ggridges)
library(styler)
library(readr)
library(dplyr)
library(janitor)
library(tidymodels)
library(mice) # mice
library(DMwR2) # KNN
library(splines) # куб инт
library(DescTools) # RobScale
library(conflicted) # конфликты

library(tidyverse)
library(scales)
library(ggrepel)
library(patchwork)
library(data.table)
library(Matrix)


# Временные ряды
library(nonlinearTseries)
library(modeltime)
library(lubridate)
library(forecast)
library(TSstudio)
library(prophet)
library(timetk)
library(zoo)
library(xts)
library(NTS)

library(fable)
library(fabletools)
library(tsibble)
library(tidyverse)
library(lubridate)


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


# Облака + карта
library(tm)
library(wordcloud)
library(RColorBrewer)
library(OpenStreetMap)


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
library(future.apply)
library(doParallel)
library(parallel)
library(foreach)
}


# Техническое 
{
num_cores = detectCores()
registerDoParallel(cores = num_cores)

# sapply(data, function(y) sum(length(which(is.na(y)))))
# windowsFonts(TNR = windowsFont("Times New Roman"))
# theme_set(theme_minimal(base_family = "TNR"))

conflicted::conflicts_prefer(dplyr::select)
conflicted::conflicts_prefer(dplyr::filter)
conflicted::conflicts_prefer(dplyr::lag)

conflicted::conflicts_prefer(base::as.matrix)
conflicts_prefer(base::intersect)
conflicts_prefer(base::setdiff)

conflicted::conflicts_prefer(lubridate::wday)
conflicts_prefer(lubridate::month)
conflicts_prefer(lubridate::year)
conflicts_prefer(zoo::index)
}




# 01: Data Filtering ------------------------------------------------------


# Импорт
import = function(directory = "data") {
  files = list.files(directory, pattern = "*.csv", full.names = TRUE)
  var_names = gsub(".csv", "", basename(files))
  
  data_list = lapply(files, function(file) {
    read_csv(file, na = c("N/A", "NA", "", ".")) |> 
      clean_names()
  })
  
  names(data_list) = var_names
  
  # Df с na
  na_counts = sapply(data_list, function(df) sum(is.na(df)))
  na_df = data.frame(dataset = names(na_counts), na_count = na_counts, row.names = NULL)
  
  # Dl в отдельные объекты
  list2env(data_list, envir = .GlobalEnv)
  
  return(na_df)
}

na_info = import("data")


# Генерация дат
dates = seq(as.Date("2013-01-01"), as.Date("2017-08-31"), by = "days") |>
  data.frame() |>
  rename(date = 1)

# Объеденение дф по дате
df_by_date = function(...) {
  dataframes = list(...)
  
  combined_df = reduce(dataframes, function(df1, df2) {
    left_join(df1, df2, by = "date")
  })
  
  return(combined_df)
}

trans = df_by_date(dates, transactions)


# Интерполяция na в oil
Oil_price = df_by_date(dates, oil)

zoo = zoo(Oil_price$dcoilwtico, Oil_price$date)
filled = na.spline(zoo)

oil_filled = data.frame(date = index(filled), oil_price = coredata(filled))
oil_filled$oil_price = round(oil_filled$oil_price, 2)


# Фильтрация holidays
holidays = holidays_events |>
  arrange(date) |>
  group_by(description) |>
  mutate(next_date = lead(date),
         next_type = lead(type),
         next_description = lead(description)) |>
  filter(!(type == "Holiday" & next_type == "Transfer" & next_description == description & next_date - date <= 7)) |>
  select(-next_date, -next_type, -next_description) |>
  ungroup()

holidays = holidays |>
  rename(day_type = type, day_locale = locale, day_transferred = transferred) |>
  mutate(day_transferred = if_else(day_transferred == "TRUE", 1, 0)) |>
  select(-locale_name, -description, -day_transferred)


# Объеденение всех дф
test = test |>
  mutate(sales = NA, data_type = "test")

train = train |>
  mutate(data_type = "train")


train_test = rbind(train, test) |>
  mutate(day_of_week = weekdays(date)) |>
  mutate(day_type = case_when(
    day_of_week %in% c("суббота", "воскесенье") ~ "Holiday",
    TRUE ~ "Work Day"
  )) |>
  select(-day_of_week)

train_test = left_join(train_test, holidays, relationship = "many-to-many", by = "date") |>
  mutate(day_type = ifelse(is.na(day_type.y), day_type.x, day_type.y)) |>
  select(-day_type.x, -day_type.y)


full_df = train_test |>
  left_join(trans, relationship = "many-to-many", by = c("date", "store_nbr")) |>
  left_join(oil_filled, relationship = "many-to-many", by = c("date")) |>
  left_join(stores, relationship = "many-to-many", by = c("store_nbr")) |>
  mutate(day_locale = replace_na(day_locale, "Ordinary")) |>
  mutate(day_week = wday(date, label = TRUE, abbr = FALSE, locale = "en_US"),
         month_year = month(date, label = TRUE, abbr = FALSE, locale = "en_US"),
         year = year(date),
         yr_month = format(date, "%Y-%m")) |>
  mutate(across(-c(id, date, sales, onpromotion, transactions, oil_price, year, yr_month), as.factor)) |>
  mutate_if(is.factor, function(x) {
    contrasts(x) = contr.sum(levels(x))
    return(x)
  }) |>
  filter(date != as.Date("2013-01-01")) |>
  select(id, date, sales, family, store_nbr, onpromotion, transactions, everything())




# 02: Exploratory Data Analysis (EDA) -------------------------------------


# Курс нефти
plot_01 = ggplot(fill(oil_filled, oil_price, .direction = "downup"),
                 aes(x = date, y = oil_price)) +
  geom_line(size = 1) +
  geom_smooth(method = "loess", se = FALSE, color = "red", linewidth = 1) +
  labs(
    x = "Года", 
    y = "Цена",
    title = "Курс нефти",
    subtitle = "Неоднородный тренд, сезонность неясна"
  ) + 
  theme(plot.title = element_text(hjust = 0),
        axis.title.x = element_blank())

dev.new()
print(plot_01)

ggsave("01.png", plot = plot_01, dpi = 300, width = 12, height = 8, units = "in")


# Объемы продаж за весь периуд
daily_transactions = transactions |>
  group_by(date) |>
  summarise(total_transactions = sum(transactions, na.rm = TRUE))

acf(daily_transactions$total_transactions, main="")
ts = ts(daily_transactions$total_transactions, start = c(2013, 1), frequency = 365)
plot(stl(ts, s.window="periodic")$time.series, main="")


plot_02 = ggplot(fill(daily_transactions, total_transactions, .direction = "downup"),
                 aes(x = date, y = total_transactions)) +
  geom_line(size = 1) +
  geom_smooth(method = "loess", se = FALSE, color = "red", linewidth = 1) +
  labs(
    x = "Года", 
    y = "Объём",
    title = "Объемы продаж",
    subtitle = "Однородный тренд, есть сезонность"
  ) + 
  theme(plot.title = element_text(hjust = 0),
        axis.title.x = element_blank())

dev.new()
print(plot_02)

ggsave("02.png", plot = plot_02, dpi = 300, width = 12, height = 8, units = "in")


# Корреляция товаров от курса нефти
sales_by_day = train |>
  group_by(family, date) |>
  summarise(sales = sum(sales))

merged = merge(sales_by_day, oil_filled, by = "date", all.x = TRUE)

corr = merged |>
  group_by(family) |>
  summarise(correlation = cor(sales, oil_price, method = "pearson")) |>
  arrange(desc(correlation))


# Тренд и сезонность
plot_03 = train |>
  group_by(date) |>
  summarise(avg_daily_sales = mean(sales)) |>
  filter(date <= "2014-01-01") |>
  ggplot() +
  geom_line(aes(x = date, y = avg_daily_sales), size = 1) +
  labs(
    x = "Даты", 
    y = "Продажи",
    title = "Общее кол-во продаж за 2013 год",
    subtitle = "и среднее под дням недели"
  ) + 
  theme(plot.title = element_text(hjust = 0),
        axis.title.x = element_blank())

plot_04 = full_df |> 
  group_by(date) |>
  summarise(avg_daily_sales = mean(sales),
            wday = day_week, .groups = "keep") |>
  mutate(wday = factor(wday, 
                          levels = c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"),
                          labels = c("Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота", "Воскресенье"))) |>
  filter(date <= "2013-04-01") |>
  ggplot()+
  geom_col(aes(x = date, y = avg_daily_sales, fill = wday))+
  scale_fill_viridis_d()+
  scale_y_continuous(labels = comma) +
  labs(x = "Дата",
       y = "Средние продажи",
       fill = "День недели") +
  theme(axis.title.x = element_blank())

comb_plot = plot_grid(plotlist = list(plot_03, plot_04), nrow = 2)
dev.new()
print(comb_plot)
ggsave("03-04.png", plot = comb_plot, dpi = 300, width = 12, height = 8, units = "in")


# Продажи
plot_05 = full_df |>
  group_by(date) |>
  filter(year == 2016, date <= "2017-01-01") |>
  ggplot() +
  geom_col(aes(x = date, y = sales, fill = type), alpha = 0.9) +
  scale_y_continuous(labels = comma) +
  labs(
    x = "Даты", 
    y = "Продажи",
    fill = "Тип",
    title = "Продажи в зависимости от типа магазина",
    subtitle = "есть корреляция"
  ) + 
  theme(plot.title = element_text(hjust = 0),
        axis.title.x = element_blank())

dev.new()
print(plot_05)

ggsave("05.png", plot = plot_05, dpi = 300, width = 12, height = 8, units = "in")




# 03: GLM transactions model ----------------------------------------------


# Разделение наборов данных
train_trans = full_df |>
  filter(data_type == "train") |>
  filter(yr_month == c("2017-05", "2017-06", "2017-07", "2017-08")) |>
  mutate(transactions = replace_na(transactions, 0))

test_trans = full_df |>
  filter(data_type == "test")

frst_train_trans = full_df |>
  filter(data_type == "train") |>
  mutate(transactions = replace_na(transactions, 0))


# Модель GLM
mod_trans = glm(log(transactions + 1) ~  store_nbr + family + center(onpromotion) + center(oil_price)
                + month_year + day_week + day_type + day_locale, 
                data = train_trans, family = gaussian(link = "identity"))
summary(mod_trans)


pred_trans = exp(predict(mod_trans, test_trans)) - 1 |>
  as.vector()


# Экспорт
test_trans[, "transactions"] = pred_trans
test_trans = test_trans |>
  group_by(date, store_nbr) |>
  mutate(transactions = round(mean(transactions))) |>
  ungroup()

sales_df = rbind(frst_train_trans, test_trans) |>
  mutate(holiday_bin = ifelse(day_type == "Holiday", 1 ,0))




# 04: Prophet + reg sales model ---------------------------------------------


# Разделение наборов данных
train_sales = sales_df |>
  filter(data_type == "train")

test_sales = sales_df |>
  filter(data_type == "test")

holidays = holidays_events |>
  filter(transferred == FALSE & type == "Holiday") |>
  select(date, locale_name) |>
  rename(ds = date, holiday = locale_name) |>
  unique()


# Многопоточность
plan(multisession, workers = availableCores() - 1)
options(future.globals.maxSize = 3 * 1024 * 1024 * 1024) # 3ГБ


# Модель
forecast_prophet = function(train_data, test_data, holidays) {
  
  # Лог sales
  train_data = train_data |>
    mutate(sales = log(sales + 1))
  
  # Prophet
  prophet_data = train_data |>
    select(date, sales, onpromotion, oil_price, transactions, holiday_bin) |>
    rename(ds = date, y = sales)
  
  # Параметры
  m = prophet(daily.seasonality = TRUE, weekly.seasonality = TRUE, yearly.seasonality = TRUE, 
              growth = "linear", seasonality.mode = "additive", holidays = holidays)
  
  # #  Регрессоры (onpromotion и transactions при lm точность сничажется)
  # m = add_regressor(m, "onpromotion")
  # m = add_regressor(m, "transactions")
  m = add_regressor(m, "holiday_bin")
  m = add_regressor(m, "oil_price")
  
  #  Модель
  m = fit.prophet(m, prophet_data)
  
  # Тестовые
  future = test_data |>
    select(date, onpromotion, oil_price, transactions, holiday_bin) |>
    rename(ds = date)
  
  # Прогнозирование
  forecast = predict(m, future)
  
  # Обратное лог
  forecast = forecast |>
    mutate(yhat = exp(yhat) - 1)
  
  # Добавление id
  return(test_data |>
           select(id) |>
           bind_cols(forecast |> 
                       select(ds, yhat) |> 
                       rename(date = ds, sales = yhat)))
}


# Прогнозирование для каждого магазина и товара
stores = unique(train_sales$store_nbr)
families = unique(train_sales$family)
all_forecasts = list()

all_forecasts = future_lapply(stores, function(store) {
  lapply(families, function(current_family) {
    train_data = train_sales |>
      filter(store_nbr == store, family == current_family)
    
    test_data = test_sales |>
      filter(store_nbr == store, family == current_family)
    
    if (nrow(test_data) > 0) {
      forecast = forecast_prophet(train_data, test_data, holidays)
      forecast$store_nbr = store
      forecast$family = current_family
      return(forecast)
    }
  })
})


# Объединение всех прогнозов
prophet_forecasts = bind_rows(all_forecasts) |>
  select(id, sales) |>
  round(digits = 3) |>
  arrange(id)

# Замена na и отр на 0
prophet_forecasts$sales[is.na(prophet_forecasts$sales)] = 0
prophet_forecasts$sales[prophet_forecasts$sales < 0] = 0

# Экспорт
write.csv(prophet_forecasts, "sales_prophet.csv", row.names = FALSE)