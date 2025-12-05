# Whats Cooking? 
# Ran for 15 min and got to .73018
library(jsonlite)
library(tidytext)
library(tidymodels)
library(tidyverse)
library(doParallel)
library(textrecipes)
library(xgboost)
library(beepr)

cl <- makePSOCKcluster(parallel::detectCores() - 1)
registerDoParallel(cl)

trainSet <- fromJSON("train.json") 
testSet <- fromJSON("test.json") 

trainSet$cuisine <- as.factor(trainSet$cuisine)

# -------------------------------------------------------------
# Feature Engineering (same as before)
# -------------------------------------------------------------
trainSet$num_ingredients <- sapply(trainSet$ingredients, length)
trainSet$avg_ingredient_length <- sapply(trainSet$ingredients, function(x) mean(nchar(x)))
trainSet$num_long_ingredients <- sapply(trainSet$ingredients, function(x) sum(sapply(strsplit(x, " "), length) > 1))

testSet$num_ingredients <- sapply(testSet$ingredients, length)
testSet$avg_ingredient_length <- sapply(testSet$ingredients, function(x) mean(nchar(x)))
testSet$num_long_ingredients <- sapply(testSet$ingredients, function(x) sum(sapply(strsplit(x, " "), length) > 1))

trainSet$num_unique_words <- sapply(trainSet$ingredients, function(x) length(unique(unlist(strsplit(x, " ")))))
testSet$num_unique_words  <- sapply(testSet$ingredients, function(x) length(unique(unlist(strsplit(x, " ")))))

trainSet$num_single_word <- sapply(trainSet$ingredients, function(x) sum(sapply(strsplit(x, " "), length) == 1))
testSet$num_single_word  <- sapply(testSet$ingredients, function(x) sum(sapply(strsplit(x, " "), length) == 1))

# -------------------------------------------------------------
# Recipe (only slight update: include numeric features)
# -------------------------------------------------------------
rec <- recipe(cuisine ~ ingredients + num_ingredients + avg_ingredient_length + num_long_ingredients + num_unique_words + num_single_word,
              data = trainSet) %>%
  step_mutate(ingredients = tokenlist(ingredients)) %>%
  step_tokenfilter(ingredients, max_tokens = 2500) %>%   # boosted from 1500
  step_tfidf(ingredients)

# -------------------------------------------------------------
# Train/validation split
# -------------------------------------------------------------
set.seed(123)
data_split <- initial_split(trainSet, prop = 0.8, strata = cuisine)
train_data <- training(data_split)
valid_data <- testing(data_split)

# -------------------------------------------------------------
# Replace RF with XGBoost (minimal edit)
# -------------------------------------------------------------
xgb_model <- boost_tree(
  trees = 230,        # More trees = better accuracy for text
  tree_depth = 7,     # Reasonable default
  learn_rate = 0.05,  # Good for sparse TF-IDF
  loss_reduction = 0, # Gamma
  sample_size = 0.8   # Subsample rows to reduce overfitting
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

# -------------------------------------------------------------
# Workflow
# -------------------------------------------------------------
wf <- workflow() %>%
  add_model(xgb_model) %>%
  add_recipe(rec)

# -------------------------------------------------------------
# Fit on training data
# -------------------------------------------------------------
xgb_fit <- wf %>% fit(data = train_data)

# -------------------------------------------------------------
# Predict on validation data
# -------------------------------------------------------------
xgb_preds <- predict(xgb_fit, valid_data) %>%
  bind_cols(valid_data %>% select(cuisine))

class_metrics <- metric_set(accuracy, kap, f_meas)

class_metrics(
  xgb_preds,
  truth = cuisine,
  estimate = .pred_class
)
beep(sound = 8)
# -------------------------------------------------------------
# Fit on full training set
# -------------------------------------------------------------
xgb_final <- wf %>% fit(data = trainSet)

# -------------------------------------------------------------
# Predict on test set
# -------------------------------------------------------------
test_predictions <- predict(xgb_final, testSet)

submission <- tibble(
  id = testSet$id,
  cuisine = test_predictions$.pred_class
)

beep(sound = 8)        # Plays the "mario" sound (sound number 8)
write_csv(submission, "submission.csv")

stopCluster(cl)
