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


# Облака
library(tm)
library(wordcloud)
library(RColorBrewer)


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
windowsFonts(TNR = windowsFont("Times New Roman"))

conflicted::conflicts_prefer(dplyr::select)
conflicted::conflicts_prefer(dplyr::filter)
conflicted::conflicts_prefer(Matrix::as.matrix)
conflicted::conflicts_prefer(base::setdiff)




# 01: Exploratory Data Analysis (EDA) ------------------------------------


# Импорт
data_eda = read_csv("data/anime.csv", na = c("N/A", "", ".", "Unknown")) |> 
  clean_names() |>
  na.omit()
View(data_eda)


# Распределение рейтинга аниме
mean_rating = mean(data_eda$rating, na.rm = TRUE)

plot_01 = ggplot(data_eda, aes(x = rating)) +
  geom_histogram(binwidth = 0.5, alpha = 0.35, fill = 'red', color = 'black') +
  geom_vline(aes(xintercept = mean_rating), color = "blue", linetype = "dashed", size = 1) +
  labs(
    x = "Рейтинг", 
    y = "Частота", 
    fill = "Тип",
    title = "Распределение рейтинга",
    caption = "Data from myanimelist.net"
  ) + 
  theme(plot.title = element_text(hjust = 0))

dev.new()
print(plot_01)


# Распределение аниме по типам
type_counts = table(data_eda$type)
sorted_types = names(type_counts)[order(type_counts, decreasing = TRUE)]
data_eda$type = factor(data_eda$type, levels = sorted_types)

plot_02 = ggplot(data_eda, aes(x = factor(type), fill = type)) +
  geom_bar(alpha = 0.5) +
  labs(
    x = "Тип", 
    y = "Количество", 
    fill = "Тип",
    title = "Распределение аниме",
    subtitle = "по типам",
    caption = "Data from myanimelist.net"
  ) + 
  theme(plot.title = element_text(hjust = 1),
        plot.subtitle = element_text(hjust = 1)) +
  geom_text(stat='count', aes(label=..count..), vjust=-1)

dev.new()
print(plot_02)


# Распределения рейтинга аниме по типам
plot_03 = ggplot(data_eda, aes(x = rating, fill = type)) +
  geom_histogram(position = "identity", alpha = 0.5, bins = 20) +
  facet_grid(. ~ type) +
  labs(
    x = "Рейтинг", 
    y = "Количество", 
    fill = "Тип",
    title = "Распределение рейтинга аниме",
    subtitle = "по типам",
    caption = "Data from myanimelist.net"
  )

dev.new()
print(plot_03)


# Разделение и группировака жанров
separated_genres = data_eda |>
  separate_rows(genre, sep = ", ") |>
  mutate(genre = trimws(genre)) |>
  filter(genre != "Na")

# Общие кол-во и %
genre_counts = table(separated_genres$genre)
genre_percentages = prop.table(genre_counts) * 100

gdf = data.frame(
  genre = names(genre_counts),
  quantity = as.numeric(genre_counts),
  percentages = as.numeric(genre_percentages),
  stringsAsFactors = FALSE
)

# Среднее
avg = separated_genres |>
  group_by(genre) |>
  summarise(members = mean(members, na.rm = TRUE),
            rating = mean(rating, na.rm = TRUE)) |>
  mutate(ratio = rating/members) |>
  arrange(desc(ratio))

# Объединение
comb_avg_gdf = cbind(gdf, avg[, c("members", "rating", "ratio")]) |>
  arrange(desc(quantity))

View(comb_avg_gdf)
head(comb_avg_gdf)

#  Корреляции для каждого жанра
correlation = comb_avg_gdf |>
  summarise(correlation = cor(members, rating, use = "complete.obs"))

plot_04 = ggplot(comb_avg_gdf, aes(members, rating)) +
  geom_point(aes(color = genre)) +
  geom_smooth(size = 1.5, se = FALSE, color = "#ffcccc") +
  labs(
    x = "Колличество зрителей", 
    y = "Рейтинг", 
    title = "Диаграмма рассеяния жанров",
    subtitle = "по рейтингу и количеству зрителей",
    caption = "Data from myanimelist.net"
  ) +
  guides(color = FALSE)

dev.new()
print(plot_04)


# Распределение рейтинга внутри жанра
filtered = separated_genres |>
  filter(genre %in% c("Comedy", "Action", "Adventure", "Fantasy", "Drama", "Sci-Fi"))

plot_05 = ggplot(filtered, aes(x = rating, fill = factor(genre))) +
  geom_histogram(binwidth = 0.5, color = "black", alpha = 0.7) +
  facet_wrap(~ genre, scales = "free") +
  labs(
    x = "Рейтинг", 
    y = "Количество", 
    fill = "Жанр",
    title = "Распределение рейтинга",
    subtitle = " внутри 6 самых популярных жанров",
    caption = "Data from myanimelist.net"
  )

dev.new()
print(plot_05)


# Корпус текста
corpus = Corpus(VectorSource(separated_genres$genre)) |>
  tm_map(content_transformer(tolower)) |>
  tm_map(removePunctuation) |>
  tm_map(removeNumbers) |>
  tm_map(removeWords, stopwords("english"))

#  Матрица терминов -> дф
dtm = TermDocumentMatrix(corpus)

word_freqs = as.data.frame(as.matrix(dtm))
word_freqs = sort(rowSums(word_freqs), decreasing=TRUE) 

# Облако 
par(mar = rep(0, 4))
plot_06 = wordcloud(names(word_freqs), freq=word_freqs, max.words=100, 
                    random.order=FALSE, colors=brewer.pal(8, "Dark2"), scale=c(8, 0.3))

dev.new()
print(plot_06)




# 02-1: Дамми переменные ---------------------------------------------------


# Импорт
anime = read_csv("data/anime.csv", na = c("N/A", "", ".", "Unknown")) |>
  clean_names() |>
  na.omit() |>
  separate_rows(genre, sep = ", ") |>
  unnest(genre) |>
  mutate(mal_rating = round(rating), members = members/100000) |>
  select(-rating)

rating = read_csv("data/rating.csv", na = c("N/A", "", ".", "-1")) |>
  clean_names() |>
  na.omit() |>
  rename(user_rating = rating)


# Создание дамми-переменных
genre_dummies = anime |>
  distinct(anime_id, genre, .keep_all = TRUE) |>
  mutate(
    value = 1,
    genre = tolower(genre))|>
  spread(key = genre, value = value, fill = 0, convert = TRUE) |>
  select(-name, -type)

type_dummies = anime |>
  distinct(anime_id, type, .keep_all = TRUE) |>
  mutate(
    value = 1,
    type = tolower(type))|>
  spread(key = type, value = value, fill = 0, convert = TRUE) |>
  select(-name, -genre, -episodes, -members, -mal_rating,)


# Общие массивы
# dummies = model.matrix(~ type + genre - 1, data = anime) - альтернатива
data_dummies = left_join(type_dummies, genre_dummies, by = "anime_id") |>
  rename(music_type = music.x, music_genre = music.y) |>
  relocate(movie, music_type, ona, ova, special,tv, .after = mal_rating)

colnames(data_dummies) = gsub(" ", "_", colnames(data_dummies))
View(data_dummies)

big_data = left_join(rating, data_dummies, by = "anime_id") |>
  filter(user_rating >=8)
data_numeric =  big_data |>
  select(-user_id, -anime_id, -episodes, -members, -mal_rating) |>
  as.data.frame()




# 02-2: IBCF - модель коллаборативной фильтрации --------------------------


# Разряженные матрицы
unique_users = unique(big_data$user_id)
unique_anime = unique(big_data$anime_id)

sparse_matrix = sparseMatrix(
  i = match(big_data$user_id, unique_users),
  j = match(big_data$anime_id, unique_anime),
  dims = c(length(unique_users), length(unique_anime)),
  dimnames = list(users = unique_users, anime = unique_anime)
)

# Сложение всех матриц вместе
for (col in colnames(data_numeric)) {
  sparse_matrix = sparse_matrix + sparseMatrix(
    i = match(big_data$user_id, unique_users),
    j = match(big_data$anime_id, unique_anime),
    x = as.numeric(data_numeric[[col]]),
    dims = c(length(unique_users), length(unique_anime)),
    dimnames = list(users = unique_users, anime = unique_anime)
  )
}

matrix = as(sparse_matrix, "realRatingMatrix")


# Разделение данных
set.seed(1234)
train_indices = sample(seq_len(nrow(matrix)), size = 6000)
test_indices = setdiff(seq_len(nrow(matrix)), train_indices)

train = matrix[train_indices, ]
test = matrix[test_indices, ]


# Модель
model = Recommender(train, method = "IBCF", parameter = list(
  k = 50, 
  alpha = 0.45, 
  na_as_zero = FALSE))

rec_df = data.frame(user_id = integer(), rec_anime_id = character())

# Перебор пользователей
for (user_id in rownames(test)) {
  users_id = test[which(rownames(test) == user_id)]
  recommendations = predict(model, users_id, n = 3)
  rec_list = as(recommendations, "list")
  
  # Добавление в дф
  if (length(rec_list[[1]]) > 0) {
    rec_df = rbind(rec_df, data.frame(user_id = user_id, rec_anime_id = paste(rec_list[[1]], collapse = ", ")))
  }
}

View(rec_df)
# write.csv(rec_df, "result.csv", row.names = FALSE)




# 03: Hybrid rec sys based on content filtering ---------------------------


# Импорт + фильтрация
af = read_csv("data/anime.csv", na = c("N/A", "", ".", "Unknown")) |>
  clean_names() |>
  na.omit() |>
  mutate(mal_rating = round(rating)) |>
  select(-rating, -members, -name)

rf = read_csv("data/rating.csv", na = c("N/A", "", ".", "-1")) |>
  clean_names() |>
  na.omit() |>
  rename(user_rating = rating) |>
  filter(user_rating >=8)

ff = left_join(rf, af, by = "anime_id")


# Ввод
user_df = ff |>
  select(-mal_rating, -user_rating) |>
  slice(1:48788)
  
anime_df = ff |>
  filter(mal_rating >=7) |>
  select(-user_id, -user_rating, -mal_rating)


# Фильтрация для пользователя
filter_anime_for_user = function(user_id, user_df, anime_df) {
  
  user_data = user_df |> 
    filter(user_id == user_id)
  
  user_genres = strsplit(user_data$genre, ",")[[1]]
  
  # Просмотренное
  watched_anime = user_df |> 
    filter(user_id == user_id) |> 
    pull(anime_id)
  
  # Если 80% и более аниме с 1 эпизодом
  if (sum(user_data$episodes == 1) / nrow(user_data) >= 0.8) {
    recommended_anime = anime_df |> 
      filter(
        str_detect(genre, paste(user_genres, collapse = "|")),
        type == user_data$type,
        episodes == 1,
        !(anime_id %in% watched_anime)
      )
  } else {
    # Если нет, то +-4 от кол-во эпизодов
    recommended_anime = anime_df |> 
      filter(
        str_detect(genre, paste(user_genres, collapse = "|")),
        type == user_data$type,
        abs(episodes - user_data$episodes) <= 4,
        !(anime_id %in% watched_anime)
      )
  }
  
  # 3 случайных
  recommended_anime = recommended_anime |> 
    sample_n(min(3, nrow(recommended_anime)))
  
  return(recommended_anime)
}


#  Для всех пользователей
rec_list = unique(user_df$user_id) |>
  map(function(user_id) {
    rec_anime = filter_anime_for_user(user_id, user_df, anime_df)
    rec_anime_ids = paste(rec_anime$anime_id, collapse = ", ")
    data.frame(user_id = user_id, rec_anime_id = rec_anime_ids)
  }) |>
  bind_rows()

View(rec_list)
# write.csv(rec_list, "result_v2_first1000.csv", row.names = FALSE)