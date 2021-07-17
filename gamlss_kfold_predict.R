library(rsample)
library(tidyverse)
library(lubridate)
library(glue)
library(gamlss)
library(gamlss.add)
library(forecast)

K=2

df_all <- bind_rows(
  read_csv("train.csv", col_types = cols()),
  read_csv("test.csv", col_types = cols())) %>%
  rename_all(
    function(col) {str_to_lower(col)})
df_train <- df_all %>% filter(!is.na(target_carbon_monoxide))
df_test  <- df_all %>% filter(is.na(target_carbon_monoxide))

# let's pretend the next year is like the first year...
df_pretend <- df_train %>%
  filter(date_time > '2010-04-04 14:00:00') %>%
  mutate(date_time = date_time + days(365))

# 399 rows without measurements
df_odd_rows <- df_all %>% 
  filter(absolute_humidity < 0.24) %>%
  transmute(date_time, no_measurement = 1)

to_train <- bind_rows(df_train, df_pretend %>% filter(date_time < '2011-06-01')) %>%
  # Don't include observations where the sensors are not working.
  filter(absolute_humidity >= 0.24) %>%
  mutate(
    hour = hour(date_time),
    wday = wday(date_time, label = FALSE),
    trend = as.numeric((as.Date(date_time) - ymd('2010-03-10')), "days"),
    working_hours = ifelse(hour(date_time) %in% 8:20, 1, 0),
    is_weekend = ifelse(wday(date_time, week_start=1)%in% 6:7, 1, 0)
  )

test <- df_test %>%
  mutate(
    hour = hour(date_time),
    wday = wday(date_time, label = FALSE),
    trend = as.numeric((as.Date(date_time) - ymd('2010-03-10')), "days"),
    working_hours = ifelse(hour(date_time) %in% 8:20, 1, 0),
    is_weekend = ifelse(wday(date_time, week_start=1)%in% 6:7, 1, 0),
  )

formula_rhs <-  ~ 
  cs(sensor_1, df=10) + cs(sensor_2, df=10) + cs(sensor_3, df=10) + 
  cs(sensor_4, df=10) + cs(sensor_5, df=10) +
  cs(deg_c, df=10) +
  cs(hour, df = 5) + 
  cs(wday, df = 5) + 
  tr(~working_hours+is_weekend)+
  ga(~s(relative_humidity,absolute_humidity), method="REML") +
  ga(~ti(hour,wday), method="REML") +
  ga(~ti(sensor_2,sensor_5), method="REML") +
  cs(trend)

# fit only based on time
formula_ts <-  ~ 
  cs(hour, df = 5) + 
  cs(wday, df = 5) + 
  cs(trend)

targets = df_all %>% colnames() %>% .[stringr::str_detect(., "target")]

set.seed(314)
logo <- to_train %>% 
  mutate(id=1:nrow(.),
         group = month(date_time)) %>% 
  group_vfold_cv(group = group, v = K)

results <- list()

for(target in targets){
  
  print(glue("Target: {target}..."))
  
  oof_pred = c()
  score = c()
  y_pred = matrix(0, nrow=nrow(test), K) %>% `names<-`(paste0("fold_", 1:K))
  
  for(i in seq_along(logo$id)) {
    
    print(glue("Fold: {i}..."))
    
    train <- rsample::analysis(logo$splits[[i]])
    valid <- rsample::assessment(logo$splits[[i]])
    
    lamb <- BoxCox.lambda(train %>% pull(target))
    
    train <- train %>% mutate_at(target, ~BoxCox(.x, lamb))
    
    fit <- gamlss(
      formula = as.formula(str_c(target,  str_c(formula_rhs, collapse = ""))),  
      sigma.formula = as.formula(str_c(formula_rhs, collapse = "")),  
      data = train, 
      family = NO,
      control = gamlss.control(#n.cyc = 400,
        trace = T)
    )
    
    fit_ts <- gamlss(
      formula = as.formula(str_c(target,  str_c(formula_ts, collapse = ""))),  
      data = to_train,
      control = gamlss.control(n.cyc = 200))
    
    
    y_val = InvBoxCox(predict(fit, newdata = valid, type = "response"), lamb)
    oof_pred[valid$id] = y_val # store oof predictions
    
    score[i] <- Metrics::rmsle(actual = valid %>% pull(target), predicted = y_val)
    
    print(glue("Fold rmlse: {score[i]}!"))
    
    y_pred[,i] = InvBoxCox(predict(fit, newdata = test, type = "response"), lamb)
    
  }
  
  print(glue("Final {target} rmlse: {mean(score)}!\n"))
  
  results[[target]] <- list(
    oof_pred = oof_pred,
    score = score,
    y_pred = y_pred %>% apply(1, mean)
  )
  
}

pred_1 = results[[targets[1]]]
pred_2 = results[[targets[2]]]
pred_3 = results[[targets[3]]]

print(pred_1$score)
print(pred_2$score)
print(pred_3$score)

print(mean(mean(pred_1$score), mean(pred_2$score), mean(pred_3$score)))

sub <- tibble(
  date_time = test$date_time,
  target_carbon_monoxide = pred_1$y_pred,
  target_benzene = pred_2$y_pred,
  target_nitrogen_oxides = pred_3$y_pred
)
