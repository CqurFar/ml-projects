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
library(mice)

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


# Моделирование
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(mlr3)
library(ranger)
library(recommenderlab)


# Многопоточность
library(doParallel)
library(parallel)
library(foreach)


# Техническое
num_cores = detectCores()
registerDoParallel(cores = num_cores)
conflicted::conflicts_prefer(dplyr::select)
conflicted::conflicts_prefer(dplyr::filter)
conflicted::conflicts_prefer(Matrix::as.matrix)




# 01: Exploratory Data Analysis (EDA) ------------------------------------


# Импорт
data = read_csv("data/titanic.csv", na = c("N/A", "", ".")) |> 
  clean_names()


# Распределение выживших и погибших по полу и классу
plot_01 = ggplot(data, aes(x = factor(survived), fill = factor(sex))) +
  geom_bar(alpha = 1/2, position = "dodge") +
  facet_grid(. ~ pclass, labeller = labeller(pclass = c("1" = "1 класс", "2" = "2 класс", "3" = "3 класс"))) +
  labs(
    x = "Статус пассажиров",
    y = "Частота",
    fill = "Пол",
    title = "Распределение выживших и погибших пассажиров",
    subtitle = "по полу и классу",
    caption = "Data from kaggle.com"
  ) +
  scale_x_discrete(labels = c("Умер", "Выжил")) +
  scale_fill_discrete(labels = c("Женский", "Мужской"))

dev.new()
print(plot_01)


# Распределение выживших и погибших по возратсу
plot_02 = ggplot(data, aes(x = age, fill = factor(survived))) +
  geom_histogram(alpha = 0.5, binwidth = 5) +
  labs(
    x = "Возраст",
    y = "Частота",
    fill = "Статус выживания",
    title = "Распределение выживших и погибших пассажиров",
    subtitle = "по возрасту",
    caption = "Data from kaggle.com"
  ) +
  scale_fill_discrete(labels = c("Умер", "Выжил"))

dev.new()
print(plot_02)

plot_03 = ggplot(data, aes(x = age, fill = factor(survived))) +
  geom_density(alpha = 0.5, color = NA) +
  labs(
    x = "Возраст",
    y = "Плотность",
    fill = "Статус выживания",
    title = "Распределение выживших и погибших пассажиров",
    subtitle = "по возрасту",
    caption = "Data from kaggle.com"
  ) +
  scale_fill_discrete(labels = c("Умер", "Выжил"))

dev.new()
print(plot_03)


# Тепловая карта
data_f = data[sapply(data, is.numeric)]
corr_matrix = cor(data_f, use = "complete.obs")
corr_df = melt(corr_matrix)

plot_04 = ggplot(data = corr_df, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile(alpha = 0.7) +
  geom_text(aes(label=round(value, 2)), size = 4) +
  scale_fill_gradient2(low = "#00bfc4", high = "#f8766d", mid = "white",
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1),
        axis.text.y = element_text(size = 12),
        axis.title.x = element_blank(),
        axis.title.y = element_blank()) +
  coord_fixed() +
  labs(
    title = "Тепловая карта корреляция", 
    subtitle = " между числовыми переменными", 
    caption = "Data from kaggle.com"
  )

dev.new()
print(plot_04)




# 02: Binomial Model ------------------------------------------------------


# Кол-во выживших и общего кол-ва пассажиров
survival_by_class = data |>
  group_by(pclass) |>
  summarise(total = n(), survived = sum(survived))

# Вероятность выживания
survival_by_class = survival_by_class |>
  mutate(prob = survived / total)

# Проверка гипотезы
binom.test(survival_by_class$survived[1], survival_by_class$total[1])
binom.test(survival_by_class$survived[3], survival_by_class$total[3])

# Результаты:
{
# 01. Вероятность выживания для пассажира третьего класса составляет 26,8% - 39,6%.
# 02. Приблизительно только 33% пассажиров третьего класса выжили.
}




# 03: SEM With Lavaan ---------------------------


mod_spec = 'survived ~ 1 + pclass + age + sib_sp + parch + fare' # модель
fmod = sem(mod_spec, data) # стуктур урав

summary(fmod)


lavaan_plot = semPaths(fmod, layout="tree2", sizeMan=7, sizeInt = 4, normalize=FALSE, 
                       whatLabels="est", edge.label.cex = 0.7, nodeLabels = names(fmod),
                       edge.color = c("#f8766d", "#00bfc4"), width=4, height=1, rotation=2, nCharNodes = 0)

# Результаты:
{
# 01. Вероятность выжить повышается на 3,9% с каждым классом.
# 02. Наличие 1 ребёнка увеличивает шансы выжить на 7,3%.
}





# 04-1: Binary Classification -------------------------------------


# Импорт
data_test = read_csv("data/test_01.csv", na = c("N/A", "", ".")) |> 
  clean_names()
View(data_test)

data_train = read_csv("data/train_01.csv", na = c("N/A", "", ".")) |> 
  clean_names()
View(data_train)


# Замена знач медианой
train = data_train |>
  mutate_if(is.numeric, ~ ifelse(is.na(.), median(., na.rm = TRUE), .))
test = data_test |>
  mutate_if(is.numeric, ~ ifelse(is.na(.), median(., na.rm = TRUE), .))


# MICE - точность ниже
{
  # # Замена знач алгоритмом MICE
  # imputed_train = mice(data_train, m=5, maxit = 50, method = 'pmm', seed = 500)
  # imputed_test = mice(data_test, m=5, maxit = 50, method = 'pmm', seed = 500)
  # 
  # # Получение окончательных наборов данных после импутации
  # train = complete(imputed_train)
  # test = complete(imputed_test)
  }


# Факторы
factors = c("pclass", "sex", "age", "sib_sp") # "parch", "fare", "embarked" - стат не значимы
train = train[, c("survived", factors)]
test = test[factors]


# Модель 
model = glm(survived ~ ., family = binomial, data = train)
summary(model)

predicted = predict(model, newdata = test, type = "response")
predicted_factor = factor(ifelse(predicted > 0.5, 1, 0), levels = c(0, 1))


# Новый датафрейм
done = data_test |>
  add_column(predicted_survived = predicted_factor, .after = 1)




# 04-2: Checking The Accuracy -------------------------------------------------


# Импорт
done = done |>
  clean_names()

titanic = read_csv("data/titanic.csv", na = c("N/A", "", ".")) |> 
  clean_names()


# Вывод 2 столбцов
comparison = data.frame(
  predicted = done$predicted_survived,
  actual = titanic$survived
)

print(comparison)


# Подсчет несоответствий
mismatches = sum(comparison$predicted != comparison$actual)

total = nrow(comparison)
per_mismatches = (mismatches / total) * 100
per_reliability = 100 - per_mismatches

cat("Несоответствия модели:", per_mismatches, "%\n")
cat("Достоверность модели:", per_reliability, "%\n")

# Выводы:
{
  # Точность модели - 94%.
}




# 04-3: Finding Outliers ----------------------------------------------------


# Множественная регрессия
modg = lm(survived ~ sex + pclass + age + sib_sp, data_train)
summary(modg) # общ стат
confint(modg) # дов инт


# Ошибоки
vif(modg)
sqrt(vif(modg)) > 2
errors = residuals(modg) # анализ остатков и ошибок
predictg = predict(modg) # предсказания значений зависимой переменной


# Графки ошибок
hist(errors)
plot(predictg, errors, xlab = "predicted", ylab = "residual")
abline(h=0,lty=3)


# Ост графики
plot(modg) # все графики
pairs(dat[,4:7]) # матрица всех попарных диаграмм рассеяния между переменными
ggpairs(dat[,4:7]) # ggpairs из GGally
outlierTest(modg) # проверка выбросов
influencePlot(modg, method = "indentity")

# Выводы:
{
  # Точность регрессионной модели 39% - низкая.
  # Выбросов немного - 3-5, на выборку в 418 чел.
  
  # Необходимо:
  # 01.    Исключить выбросы - исключил, оказалось стат не значимо.
  # 02.    Добавить новые признаки из уже существующих (инжиниринг признаков).
  # 03.    Использовать ансамблевые методы.
  # 04.    Провести кросс-валидацию.
}
  
  


# 05-1: Random Forest -----------------------------------------------------

  
  # Модель
  rf_model = randomForest(as.factor(survived) ~ ., data = train, ntree = 1000, mtry = 0.8, sampsize = 200)
  rf_predictions = predict(rf_model, newdata = test)
  
  
  # Сравнение предсказанных и фактических значений
  comparison_rf = data.frame(
    predicted = as.numeric(as.character(rf_predictions)),
    actual = titanic$survived
  )
  
  
  # Подсчет несоответствий
  mismatches_rf = sum(comparison_rf$predicted != comparison_rf$actual)
  
  total_rf = nrow(comparison_rf)
  per_mismatches_rf = (mismatches_rf / total_rf) * 100
  per_reliability_rf = 100 - per_mismatches_rf
  
  cat("Несоответствия модели случайного леса:", per_mismatches_rf, "%\n")
  cat("Достоверность модели случайного леса:", per_reliability_rf, "%\n")
  
  
  # Сохранение результатов
  {
    # # Экспорт
    # final = data_test |>
    #   add_column(predicted_survived = rf_predictions, .after = 1)
    # write.csv(final, "final.csv")
  }
  
  # Выводы:
  {
    # Точность модели - 95%.
  }
  
  
  

# 05-2: Cross-Validation ---------------------------------------------------
  
  
  # Подстройка параметров
  control = trainControl(method = "cv", number = 10)
  model_cv = train(as.factor(survived) ~ ., data = train, method = "rf", trControl = control)
  print(model_cv)
  
  # Выводы:
  {
    # При mtry = 0.8 и  sampsize = 200 точность повыселась до 98,32%.
  }
  


# ML Results ------------------------------------------------
{
  # Точность предсказания составила - 98,32%.
}