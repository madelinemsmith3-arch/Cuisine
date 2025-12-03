# setwd("C:\\Users\\madel\\OneDrive\\Documents\\Stat 348\\Cuisine")
library(jsonlite)
library(readr)
library(dplyr)
library(tidyr)
library(tidytext)
library(stringr)
library(tidymodels)
library(textrecipes)
library(stopwords)

trainSet <- read_file("train.json") %>%
  fromJSON()
testSet <- read_file("test.json") %>%
  fromJSON()
names(trainSet)
View(trainSet)

train_long <- trainSet %>%
  unnest(ingredients)

test_long <- testSet %>%
  unnest(ingredients)

clean_tokens <- train_tokens %>%
  anti_join(stop_words, by = "word")

word_counts <- clean_tokens %>%
  count(word, sort = TRUE)

spice_features <- trainSet %>%
  unnest(ingredients) %>%
  mutate(word = tolower(ingredients)) %>%
  group_by(id, cuisine) %>%
  summarise(
    has_cumin    = any(str_detect(word, "cumin")),
    has_basil    = any(str_detect(word, "basil")),
    has_soy      = any(str_detect(word, "soy")),
    has_tumeric  = any(str_detect(word, "turmeric|tumeric")),
    has_masala   = any(str_detect(word, "masala")),
    has_cilantro = any(str_detect(word, "cilantro")),
    has_tortilla = any(str_detect(word, "tortilla")),
    has_ginger   = any(str_detect(word, "ginger")),
    has_sesame   = any(str_detect(word, "sesame")),
    has_parmesan = any(str_detect(word, "parmesan")),
    has_olive    = any(str_detect(word, "olive")),
    .groups = "drop"
  )

spice_test <- testSet %>%
  unnest(ingredients) %>%
  mutate(word = tolower(ingredients)) %>%
  group_by(id) %>%
  summarise(
    has_cumin    = any(str_detect(word, "cumin")),
    has_basil    = any(str_detect(word, "basil")),
    has_soy      = any(str_detect(word, "soy")),
    has_tumeric  = any(str_detect(word, "turmeric|tumeric")),
    has_masala   = any(str_detect(word, "masala")),
    has_cilantro = any(str_detect(word, "cilantro")),
    has_tortilla = any(str_detect(word, "tortilla")),
    has_ginger   = any(str_detect(word, "ginger")),
    has_sesame   = any(str_detect(word, "sesame")),
    has_parmesan = any(str_detect(word, "parmesan")),
    has_olive    = any(str_detect(word, "olive")),
    .groups = "drop"
  )

train_joined <- left_join(trainSet, spice_features, by = c("id","cuisine"))
test_joined  <- left_join(testSet,  spice_test,    by = "id")

train_joined <- train_joined %>%
  mutate(text = sapply(ingredients, function(x) paste(x, collapse = " ")))

test_joined <- test_joined %>%
  mutate(text = sapply(ingredients, function(x) paste(x, collapse = " ")))

my_recipe <- recipe(cuisine ~ text + has_cumin + has_basil + has_soy +
                      has_tumeric + has_masala + has_cilantro +
                      has_tortilla + has_ginger + has_sesame +
                      has_parmesan + has_olive,
                    data = train_joined) %>%
  step_tokenize(text) %>%
  step_stopwords(text) %>% 
  step_tokenfilter(text, max_tokens = 2000) %>%
  step_tf(text)


my_model <- rand_forest(
  mtry = 30,
  trees = 500,
  min_n = 10
) %>%
  set_engine("ranger") %>%
  set_mode("classification")

my_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_model)

fit_model <- my_wf %>%
  fit(data = train_joined)

preds <- predict(fit_model, test_joined) %>%
  bind_cols(test_joined %>% select(id))

submission <- preds %>%
  rename(cuisine = .pred_class)

write_csv(submission, "regression_trees.csv")

