# libraries
library(tidyverse)
library(tidymodels)
library(vroom)
library(poissonreg)
library(stacks)
library(embed)
library(doParallel)

# Prepare parallelization
num_cores <- 4

# Read data
train_dirty <- vroom("train.csv") %>%
  mutate(across(starts_with("cat"), factor))
test_dirty <- vroom("test.csv") %>%
  mutate(across(starts_with("cat"), factor))

################################
# Penalized Poisson Regression #
################################

# Recipe
pois_recipe <- recipe(loss ~ ., data = train_dirty) %>%
  step_interact(terms = ~ all_numeric_predictors()^2) %>%
  step_lencode_glm(all_factor_predictors(), outcome = vars(loss)) %>%
  step_normalize(all_predictors())

# Model
pois_model <- poisson_reg(penalty = tune(), mixture = tune()) %>%
  set_mode("regression") %>%
  set_engine("glmnet")

# Workflow
pois_workflow <- workflow() %>%
  add_model(pois_model) %>%
  add_recipe(pois_recipe)

# Cross validate
pois_folds <- vfold_cv(train_dirty, v = 20, repeats = 1)
pois_param_grid <- grid_regular(penalty(), mixture(), levels = 5)
num_cores <- 4
cl <- makePSOCKcluster(num_cores)
registerDoParallel(cl)
pois_cv <- pois_workflow %>%
  tune_grid(resamples = pois_folds,
            grid = pois_param_grid,
            metrics = metric_set(rmse, mae),
            control = control_stack_grid())
stopCluster(cl)
pois_besttune <- pois_cv %>%
  select_best(metric = "mae")

# Fit model
pois_fit <- pois_workflow %>%
  finalize_workflow(pois_besttune) %>%
  fit(data = train_dirty)
pois_pred <- predict(pois_fit, new_data = test_dirty)$.pred

# Write output
tibble(id = test_dirty$id,
       loss = pois_pred) %>%
  vroom_write("penalized_poisson.csv", delim = ",")

###############################
# Penalized Linear Regression #
###############################

log_train_dirty <- train_dirty %>%
  mutate(loss = log(loss))

# Recipe
log_recipe <- recipe(loss ~ ., data = log_train_dirty) %>%
  step_interact(terms = ~ cat1:cat15 +
                  cat1:cat22 +
                  cat1:cat62 +
                  cat1:cat64 +
                  cat1:cat70 +
                  cat2:cat70 +
                  cat3:cat8 +
                  cat3:cat66 +
                  cat3:cat70 +
                  cat5:cat70 +
                  cat6:cat8 +
                  cat6:cat9 +
                  cat6:cat10 +
                  cat6:cat11 +
                  cat6:cat12 +
                  cat6:cat13 +
                  cat6:cat16 +
                  cat6:cat19 +
                  cat6:cat21 +
                  cat6:cat23 +
                  cat6:cat24 +
                  cat6:cat25 +
                  cat6:cat26 +
                  cat6:cat27 +
                  cat6:cat28 +
                  cat6:cat29 +
                  cat6:cat30 +
                  cat6:cat31 +
                  cat6:cat32 +
                  cat6:cat33 +
                  cat6:cat34 +
                  cat6:cat35 +
                  cat6:cat36 +
                  cat6:cat37 +
                  cat6:cat38 +
                  cat6:cat39 +
                  cat6:cat40 +
                  cat6:cat41 +
                  cat6:cat42 +
                  cat6:cat43 +
                  cat6:cat44 +
                  cat6:cat45 +
                  cat6:cat46 +
                  cat6:cat47 +
                  cat6:cat48 +
                  cat6:cat61 +
                  cat6:cat66 +
                  cat6:cat69 +
                  cat6:cat70 +
                  cat8:cat16 +
                  cat8:cat17 +
                  cat8:cat18 +
                  cat8:cat19 +
                  cat8:cat61 +
                  cat9:cat70 +
                  cat13:cat56 +
                  cat15:cat17 +
                  cat15:cat18 +
                  cat15:cat24 +
                  cat15:cat36 +
                  cat15:cat41 +
                  cat15:cat54 +
                  cat15:cat65 +
                  cat15:cat66 +
                  cat15:cat67 +
                  cat17:cat18 +
                  cat17:cat19 +
                  cat17:cat56 +
                  cat18:cat56 +
                  cat19:cat35 +
                  cat19:cat58 +
                  cat19:cat70 +
                  cat20:cat57 +
                  cat21:cat35 +
                  cat21:cat50 +
                  cat21:cat66 +
                  cat21:cat71 +
                  cat23:cat70 +
                  cat27:cat31 +
                  cat27:cat50 +
                  cat32:cat59 +
                  cat32:cat64 +
                  cat32:cat70 +
                  cat33:cat62 +
                  cat34:cat35 +
                  cat35:cat61 +
                  cat35:cat65 +
                  cat35:cat68 +
                  cat37:cat38 +
                  cat37:cat39 +
                  cat37:cat44 +
                  cat37:cat49 +
                  cat37:cat50 +
                  cat37:cat53 +
                  cat37:cat70 +
                  cat38:cat55 +
                  cat38:cat69 +
                  cat39:cat44 +
                  cat39:cat46 +
                  cat39:cat50 +
                  cat39:cat69 +
                  cat40:cat45 +
                  cat41:cat45 +
                  cat42:cat60 +
                  cat42:cat70 +
                  cat43:cat45 +
                  cat48:cat51 +
                  cat48:cat55 +
                  cat48:cat56 +
                  cat48:cat68 +
                  cat49:cat50 +
                  cat49:cat52 +
                  cat49:cat53 +
                  cat50:cat51 +
                  cat50:cat52 +
                  cat50:cat53 +
                  cat50:cat54 +
                  cat50:cat55 +
                  cat50:cat56 +
                  cat50:cat61 +
                  cat50:cat66 +
                  cat50:cat67 +
                  cat60:cat68 +
                  cat50:cat69 +
                  cat50:cat70 +
                  cat51:cat55 +
                  cat51:cat56 +
                  cat51:cat67 +
                  cat52:cat53 +
                  cat53:cat60 +
                  cat54:cat62 +
                  cat54:cat70 +
                  cat56:cat59 +
                  cat59:cat62 +
                  cat60:cat62 +
                  cat61:cat65 +
                  cat61:cat66 +
                  cat65:cat70 +
                  cat66:cat70 +
                  cat67:cat68 +
                  cat67:cat70 +
                  cat70:cat72) %>%
  step_lencode_glm(all_factor_predictors(), outcome = vars(loss)) %>%
  step_normalize(all_predictors())

# Model
log_model <- linear_reg(penalty = tune(), mixture = tune()) %>%
  set_mode("regression") %>%
  set_engine("glmnet")

# Workflow
log_workflow <- workflow() %>%
  add_model(log_model) %>%
  add_recipe(log_recipe)

# Cross validate
log_folds <- vfold_cv(log_train_dirty, v = 20, repeats = 1)
log_param_grid <- grid_regular(penalty(), mixture(), levels = 5)
cl <- makePSOCKcluster(num_cores)
registerDoParallel(cl)
log_cv <- log_workflow %>%
  tune_grid(resamples = log_folds,
            grid = log_param_grid,
            metrics = metric_set(rmse, mae),
            control = control_stack_grid())
stopCluster(cl)
log_besttune <- log_cv %>%
  select_best(metric = "mae")

# Fit model
log_fit <- log_workflow %>%
  finalize_workflow(log_besttune) %>%
  fit(data = log_train_dirty)
log_pred <- predict(log_fit, new_data = test_dirty)$.pred %>%
  exp()

# Write output
tibble(id = test_dirty$id,
       loss = log_pred) %>%
  vroom_write("penalized_linear.csv", delim = ",")

#########################
# Zero-Inflated Poisson #
#########################

#################
# Random Forest #
#################

# Recipe
forest_recipe <- recipe(loss ~ ., data = train_dirty) %>%
  step_dummy(all_factor_predictors())

# Model
forest_model <- rand_forest(mtry = tune(),
                            min_n = tune(),
                            trees = 100) %>%
  set_engine("ranger") %>%
  set_mode("regression")

# Workflow
forest_workflow <- workflow() %>%
  add_model(forest_model) %>%
  add_recipe(forest_recipe)

# Cross validate
forest_folds <- vfold_cv(train_dirty, v = 10, repeats = 1)
forest_param_grid <- grid_regular(mtry(range = c(1, 11)), min_n(), levels = 20)
cl <- makePSOCKcluster(num_cores)
registerDoParallel(cl)
forest_cv <- forest_workflow %>%
  tune_grid(resamples = forest_folds,
            grid = forest_param_grid,
            metrics = metric_set(rmse, mae),
            control = control_stack_grid())
stopCluster(cl)
forest_besttune <- forest_cv %>%
  select_best(metric = "mae")

# Fit model
forest_fit <- forest_workflow %>%
  finalize_workflow(forest_besttune) %>%
  fit(data = train_dirty)
forest_pred <- predict(forest_fit, new_data = test_dirty)$.pred

tibble(id = test_dirty$id,
       loss = forest_pred) %>%
  vroom_write("random_forest.csv", delim = ",")


#################
# Boosted Trees #
#################

# Recipe
boost_recipe <- recipe(loss ~ ., data = log_train_dirty) %>%
  step_interact(terms = ~ cat1:cat15 +
                  cat1:cat22 +
                  cat1:cat62 +
                  cat1:cat64 +
                  cat1:cat70 +
                  cat2:cat70 +
                  cat3:cat8 +
                  cat3:cat66 +
                  cat3:cat70 +
                  cat5:cat70 +
                  cat6:cat8 +
                  cat6:cat9 +
                  cat6:cat10 +
                  cat6:cat11 +
                  cat6:cat12 +
                  cat6:cat13 +
                  cat6:cat16 +
                  cat6:cat19 +
                  cat6:cat21 +
                  cat6:cat23 +
                  cat6:cat24 +
                  cat6:cat25 +
                  cat6:cat26 +
                  cat6:cat27 +
                  cat6:cat28 +
                  cat6:cat29 +
                  cat6:cat30 +
                  cat6:cat31 +
                  cat6:cat32 +
                  cat6:cat33 +
                  cat6:cat34 +
                  cat6:cat35 +
                  cat6:cat36 +
                  cat6:cat37 +
                  cat6:cat38 +
                  cat6:cat39 +
                  cat6:cat40 +
                  cat6:cat41 +
                  cat6:cat42 +
                  cat6:cat43 +
                  cat6:cat44 +
                  cat6:cat45 +
                  cat6:cat46 +
                  cat6:cat47 +
                  cat6:cat48 +
                  cat6:cat61 +
                  cat6:cat66 +
                  cat6:cat69 +
                  cat6:cat70 +
                  cat8:cat16 +
                  cat8:cat17 +
                  cat8:cat18 +
                  cat8:cat19 +
                  cat8:cat61 +
                  cat9:cat70 +
                  cat13:cat56 +
                  cat15:cat17 +
                  cat15:cat18 +
                  cat15:cat24 +
                  cat15:cat36 +
                  cat15:cat41 +
                  cat15:cat54 +
                  cat15:cat65 +
                  cat15:cat66 +
                  cat15:cat67 +
                  cat17:cat18 +
                  cat17:cat19 +
                  cat17:cat56 +
                  cat18:cat56 +
                  cat19:cat35 +
                  cat19:cat58 +
                  cat19:cat70 +
                  cat20:cat57 +
                  cat21:cat35 +
                  cat21:cat50 +
                  cat21:cat66 +
                  cat21:cat71 +
                  cat23:cat70 +
                  cat27:cat31 +
                  cat27:cat50 +
                  cat32:cat59 +
                  cat32:cat64 +
                  cat32:cat70 +
                  cat33:cat62 +
                  cat34:cat35 +
                  cat35:cat61 +
                  cat35:cat65 +
                  cat35:cat68 +
                  cat37:cat38 +
                  cat37:cat39 +
                  cat37:cat44 +
                  cat37:cat49 +
                  cat37:cat50 +
                  cat37:cat53 +
                  cat37:cat70 +
                  cat38:cat55 +
                  cat38:cat69 +
                  cat39:cat44 +
                  cat39:cat46 +
                  cat39:cat50 +
                  cat39:cat69 +
                  cat40:cat45 +
                  cat41:cat45 +
                  cat42:cat60 +
                  cat42:cat70 +
                  cat43:cat45 +
                  cat48:cat51 +
                  cat48:cat55 +
                  cat48:cat56 +
                  cat48:cat68 +
                  cat49:cat50 +
                  cat49:cat52 +
                  cat49:cat53 +
                  cat50:cat51 +
                  cat50:cat52 +
                  cat50:cat53 +
                  cat50:cat54 +
                  cat50:cat55 +
                  cat50:cat56 +
                  cat50:cat61 +
                  cat50:cat66 +
                  cat50:cat67 +
                  cat60:cat68 +
                  cat50:cat69 +
                  cat50:cat70 +
                  cat51:cat55 +
                  cat51:cat56 +
                  cat51:cat67 +
                  cat52:cat53 +
                  cat53:cat60 +
                  cat54:cat62 +
                  cat54:cat70 +
                  cat56:cat59 +
                  cat59:cat62 +
                  cat60:cat62 +
                  cat61:cat65 +
                  cat61:cat66 +
                  cat65:cat70 +
                  cat66:cat70 +
                  cat67:cat68 +
                  cat67:cat70 +
                  cat70:cat72) %>%
  step_lencode_glm(all_factor_predictors(), outcome = vars(loss))

# Model
boost_model <- boost_tree(tree_depth = tune(),
                          trees = 100,
                          learn_rate = tune()) %>%
  set_mode("regression") %>%
  set_engine("xgboost")
boost_model <- boost_tree(tree_depth = 12,
                          trees = 500,
                          learn_rate = 0.0325) %>%
  set_mode("regression") %>%
  set_engine("xgboost")
# Workflow
boost_workflow <- workflow() %>%
  add_recipe(boost_recipe) %>%
  add_model(boost_model)

# Cross validate
boost_folds <- vfold_cv(log_train_dirty, v = 20, repeats = 1)
boost_param_grid <- grid_regular(tree_depth(), learn_rate(), levels = 10)
cl <- makePSOCKcluster(num_cores)
registerDoParallel(cl)
boost_cv <- boost_workflow %>%
  tune_grid(resamples = boost_folds,
            grid = boost_param_grid,
            metrics = metric_set(rmse, mae),
            control = control_stack_grid())
stopCluster(cl)
boost_besttune <- boost_cv %>%
  select_best(metric = "mae")

# Fit model
boost_fit <- boost_workflow %>%
  finalize_workflow(boost_besttune) %>%
  fit(data = log_train_dirty)
boost_pred <- predict(boost_fit, new_data = test_dirty)$.pred %>%
  exp()

tibble(id = test_dirty$id,
       loss = boost_pred) %>%
  vroom_write("boosted_trees.csv", delim = ",")

