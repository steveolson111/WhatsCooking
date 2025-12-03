# Whats Cooking? Aiming for .72
library(jsonlite)
library(tidytext)
library(tidymodels)
library(doParallel)

cl <- makePSOCKcluster(parallel::detectCores() - 1)
registerDoParallel(cl)


trainSet <- read_file("train.json") %>%
  fromJSON()
testSet <- read_file("test.json") %>%
  fromJSON()

trainSet$cuisine <- as.factor(trainSet$cuisine)


dim(trainSet)
names(trainSet)
class(trainSet$ingredients)
trainSet$ingredients[[1]]

trainSet$num_ingredients <- sapply(trainSet$ingredients, length)
trainSet$avg_ingredient_length <- sapply(trainSet$ingredients, function(x) mean(nchar(x)))
trainSet$num_long_ingredients <- sapply(trainSet$ingredients, function(x) sum(sapply(strsplit(x, " "), length) > 1))

testSet$num_ingredients <- sapply(testSet$ingredients, length)
testSet$avg_ingredient_length <- sapply(testSet$ingredients, function(x) mean(nchar(x)))
testSet$num_long_ingredients <- sapply(testSet$ingredients, function(x) sum(sapply(strsplit(x, " "), length) > 1))

## Define TF-IDF
rec <- recipe(cuisine ~ ingredients, data = trainSet) %>%
  step_mutate(ingredients = tokenlist(ingredients)) %>%
  step_tokenfilter(ingredients, max_tokens=1500) %>%
  step_tfidf(ingredients)

set.seed(123)
data_split <- initial_split(trainSet, prop = 0.8, strata = cuisine)
train_data <- training(data_split)
valid_data <- testing(data_split)

rf_model <- rand_forest(
  mtry = 50,        # number of predictors to try at each split
  trees = 200,      # number of trees
  min_n = 10         # minimum node size
) %>%
  set_engine("ranger") %>%
  set_mode("classification")

wf <- workflow() %>%
  add_model(rf_model) %>%
  add_recipe(rec)

rf_fit <- wf %>% fit(data = train_data)

rf_preds <- predict(rf_fit, valid_data) %>%
  bind_cols(valid_data %>% select(cuisine))

class_metrics <- metric_set(accuracy, kap, f_meas)

class_metrics(
  rf_preds,
  truth = cuisine,
  estimate = .pred_class)

rf_final <- wf %>% fit(data = trainSet)

test_predictions <- predict(rf_final, testSet)

head(test_predictions)

submission <- tibble(
  id = testSet$id,
  cuisine = test_predictions$.pred_class
)

write_csv(submission, "submission.csv")

stopCluster(cl)
