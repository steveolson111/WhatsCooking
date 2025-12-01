# Whats Cooking?
library(jsonlite)
library(tidyverse)
library(tidytext)

# ----------------------------
# 1. Load data
# ----------------------------
trainSet <- fromJSON("train.json")
testSet  <- fromJSON("test.json")

# ----------------------------
# 2. Make tidy long dataset 
#    (1 ingredient per row)
# ----------------------------
train_tidy <- trainSet %>%
  unnest_longer(ingredients) %>%     # expands list of ingredients
  rename(token = ingredients)        # rename for clarity (optional)

# ----------------------------
# 3. Feature 1: number of ingredients
# ----------------------------
feature_num_ingredients <- trainSet %>%
  mutate(num_ingredients = map_int(ingredients, length)) %>%
  select(id, num_ingredients)

# ----------------------------
# 4. Feature 2: TF-IDF features
# ----------------------------
tfidf_features <- train_tidy %>%
  count(id, token) %>%               # count ingredient frequencies
  bind_tf_idf(term = token,
              document = id,
              n = n)

# ----------------------------
# 5. Feature 3: ingredient presence flags
# ----------------------------
ingredient_flags <- train_tidy %>%
  distinct(id, token) %>%       # remove duplicates
  mutate(value = 1) %>%
  pivot_wider(
    id_cols = id,
    names_from = token,
    values_from = value,
    values_fill = 0)
#Combined features...

