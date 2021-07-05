
library(tidymodels)
library(tidyverse)
library(anomalize)
library(patchwork)
library(fpp3)
library(treesnip)

theme_set(theme_bw())

doParallel::registerDoParallel()

train <- readr::read_csv('train.csv')
test <-  readr::read_csv('test.csv')
sub <-   readr::read_csv('sample_submission.csv')

prepare_data <- function(x){
  x %>%
    mutate(date_time = lubridate::ymd_hms(date_time))%>%
    rename(date = date_time)
}

get_target <- function(x, to_remove=NULL){
  
  x = x %>% 
    select(-one_of(to_remove))
  
  if(!is.null(to_remove)){
    colnames(x)[ncol(x)] = "target"        
  }
  x 
}

fix_anomalies <- function(train, anom_target, order=10){
  # Before fix anomalies
  g <- anomalies[[which(X_colnames==anom_target)]] %>%
    time_recompose() %>%
    plot_anomalies(time_recomposed = TRUE, ncol = 3, alpha_dots = 0.5) +
    labs(y = NULL, title = glue::glue("{anom_target} - Before fix anomalies"))
  print(g)
  
  # Define formula
  to_remove <- c(c('date', anom_target), colnames(train) %>% .[str_detect(., "target")])
  x_form    <- paste0(colnames(train) %>% .[!.%in%to_remove], collapse = "+")
  form_lm   <- as.formula(glue::glue("{anom_target} ~ {x_form} + season('day', {order}, type = 'additive')"))
  #form_lm   <- as.formula(paste0("anom_target~", x_form, "+trend()+season()"))
  print(form_lm)
  
  # Fit prophet
  fit1 <- train %>% 
    as_tsibble(index = date) %>% 
    model(tslm = fable.prophet::prophet(form_lm))
  #model(tslm = TSLM(form_lm))
  
  # See model
  g <- augment(fit1) %>%
    ggplot(aes(x = date)) +
    geom_line(aes(y = !!as.name(anom_target), colour = "Data")) +
    geom_line(aes(y = .fitted, colour = "Fitted"), alpha=0.5) +
    labs(y = NULL, title = glue::glue("{anom_target} - Model"))+
    scale_colour_manual(values=c(Data="black",Fitted="#D55E00")) +
    guides(colour = guide_legend(title = NULL))+
    theme(legend.position = 'bottom')
  
  print(g)
  
  train[train$date%in%anomalies_idx[[anom_target]], anom_target] <- 
    augment(fit1)[augment(fit1)$date%in%anomalies_idx[[anom_target]], '.fitted']
  
  # After fix anomalies
  g <- train %>% 
    time_decompose(anom_target) %>%
    anomalize(remainder, method="gesd") %>%
    time_recompose() %>%
    plot_anomalies(time_recomposed = TRUE, ncol = 3, alpha_dots = 0.5) +
    labs(y = NULL, title = glue::glue("{anom_target} - After fix anomalies"))
  print(g)
  
  return(train)
  
}


cv_fit_predict <- function(train, test){
  
  folds_train      <- vfold_cv(train, v = 5)
  folds_test <- vfold_cv(test, v = 5) %>% rename(splits_test = splits)
  folds      <- left_join(folds_test, folds_train, by = "id") %>% 
    select(id, everything())
  
  recipe_base <- recipe(target ~ ., train)%>%
    #step_lag(deg_C, relative_humidity, absolute_humidity, 
    #         sensor_1, sensor_2, sensor_3, sensor_4,
    #         sensor_5, lag = c(1, 2, 7, 15, 30), default=0) %>%
    timetk::step_timeseries_signature(date) %>%
    step_rm(matches("(.iso$)|(.xts$)|(minute)|(second)|(date_month)|(date_wday.lbl)")) %>%
    step_rm(date) %>%
    MachineShop::step_kmeans(deg_C, relative_humidity, absolute_humidity, 
                             sensor_1, sensor_2, sensor_3, sensor_4,
                             sensor_5, k = 5, replace=FALSE)
  #step_ns(deg_C:sensor_5, deg_free = 20)
  
  model_spec_xgb <- boost_tree(
    trees = 500,            # num_iterations
    learn_rate = 0.01,      # eta
    min_n = tune(),         # min_data_in_leaf
    tree_depth = tune(),    # max_depth
    sample_size = tune(),   # bagging_fraction
    mtry = tune(),          # feature_fraction
    loss_reduction = tune() # min_gain_to_split
  ) %>% 
    set_engine("lightgbm", 
               reg_lambda = tune("reg_lambda"),
               reg_alpha = tune("reg_alpha"),
               num_leaves = tune("num_leaves"),
               #num_leaves = 31,
               early_stop = 50,
               validation = .10,
               eval_metric = "rmse"
    ) %>% 
    set_mode("regression")
  
  wf <- workflow() %>%
    add_model(model_spec_xgb) %>%
    add_recipe(recipe_base)
  
  xgb_params <- dials::parameters(
    # learn_rate(),           # learning_rate
    # trees()                 # num_iterations
    min_n(),                  # min_data_in_leaf
    tree_depth(c(1, 7)),     # max_depth
    sample_prop(c(0.1, 0.9)), # bagging_fraction (to sample_size)
    mtry(),                   # feature_fraction
    loss_reduction(),         # min_gain_to_split
    reg_lambda = dials::penalty(),
    reg_alpha = dials::penalty(),
    num_leaves = dials::tree_depth(c(15, 255))
  )
  
  xgb_params <- xgb_params %>% 
    finalize(train)
  
  set.seed(321)
  xgboost_tune <-
    wf %>%
    tune_bayes(
      resamples = folds_train,
      param_info = xgb_params,
      initial = 15,
      iter = 2, 
      metrics = metric_set(rmse),
      control = control_bayes(no_improve = 10, 
                              save_pred = F,
                              verbose = T)
    )
  
  best_model <- select_best(xgboost_tune, "rmsle")
  print(best_model)
  
  model_spec_xgb_final <- finalize_model(model_spec_xgb, best_model)
  wf <- wf %>% update_model(model_spec_xgb_final)
  
  res <- folds %>% 
    mutate_at(c('splits_test', 'splits'), ~map(.x, assessment)) %>% 
    mutate(model_fit = map(splits, ~fit(wf, data=.x))) %>% 
    mutate(y_oof = map2(model_fit, splits, ~predict(.x, .y))) %>% 
    mutate(y_pred = map2(model_fit, splits_test, ~predict(.x, .y)))
  
  return(res)
  
}

# Code --------------------------------------------------------------------



test = test %>% prepare_data()
train = train %>% prepare_data()

X_colnames <- c("deg_C", "relative_humidity", "absolute_humidity", paste0("sensor_", 1:5))

anomalies <- map(X_colnames, ~{
  train %>% 
    time_decompose(.x) %>%
    anomalize(remainder, method="gesd")
})

anomalies_idx <- map(anomalies, ~.x %>%
                       time_recompose() %>% 
                       filter(anomaly == 'Yes') 
                     %>% pull(date)) %>% 
  `names<-`(X_colnames)

y_colnames <- c("target_carbon_monoxide", "target_benzene", "target_nitrogen_oxides")

anomalies_y <- map(y_colnames, ~{
  train %>% 
    time_decompose(.x) %>%
    anomalize(remainder, method= "gesd")
})

train_fixed <- fix_anomalies(train, 'deg_C', order=5)
train_fixed <- fix_anomalies(train, 'relative_humidity', order=5)
train_fixed <- fix_anomalies(train, 'absolute_humidity', order=5)
train_fixed <- fix_anomalies(train, 'sensor_3', order=3)
train_fixed <- fix_anomalies(train, 'sensor_4', order=3)
train_fixed <- fix_anomalies(train, 'sensor_5', order=5)


train1 = get_target(train_fixed, c('target_benzene', 'target_nitrogen_oxides'))
train2 = get_target(train_fixed, c('target_carbon_monoxide','target_nitrogen_oxides'))
train3 = get_target(train_fixed, c('target_carbon_monoxide', 'target_benzene'))

res1 <- cv_fit_predict(train1, test)
res2 <- cv_fit_predict(train2, test)
res3 <- cv_fit_predict(train3, test)

y_pred <- map(list(res1, res2, res3),~{
  .x %>% 
    select(splits_test, ,y_pred) %>% 
    unnest(cols = c(splits_test, y_pred)) %>% 
    select(date, .pred) %>% 
    arrange(date) %>% 
    pull(.pred)
})


sub <- bind_cols(select(sub, date_time),
                 tibble(target_carbon_monoxide = y_pred[[1]]), 
                 tibble(target_benzene         = y_pred[[2]]),
                 tibble(target_nitrogen_oxides = y_pred[[3]]))


sub


