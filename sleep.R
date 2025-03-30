# ========================================================================================================
# Purpose:      Rscript for Computer Based Assignment CBA
# Author:       Anthony Koh Hong Ji
# Updated:      12-04-2024
# Topics:       Linear Regression; Random Forest; MARS
# Data Source:  sleep.csv
# Packages:     
#=========================================================================================================
# LIBRARY ------------------------------------------------------------------------------------------------
library(janitor)
library(visdat)
library(ggplot2)
library(corrplot)
library(doBy)
library(caTools)
library(randomForest)
library(earth)

# IMPORTING OF DATASET -----------------------------------------------------------------------------------
setwd("C:/Users/Anthony/Documents/NTU/Y2S2/BC2407 Course Materials/CBA Paper")
sleep <- read.csv("sleep.csv")
sleep <- clean_names(sleep)
sleep <- sleep[,-1]

# DATA CLEANING ----------------------------------------------------------------------
# Checking for missing values
vis_miss(sleep)

# Changing 'gender' and 'smoking_status' datatype
sleep$gender <- factor(sleep$gender)
sleep$smoking_status <- factor(sleep$smoking_status)

# Extracting 'bedtime' and 'wakeup_time' hour & minutes
sleep$bedtime <- as.POSIXct(sleep$bedtime, format = "%m/%d/%Y %H:%M")
sleep$bedtime <- factor(format(sleep$bedtime, "%H:%M"))
sleep$wakeup_time <- as.POSIXct(sleep$wakeup_time, format = "%m/%d/%Y %H:%M")
sleep$wakeup_time <- factor(format(sleep$wakeup_time, "%H:%M"))

# Splitting into age groups
age_breaks <- c(9, 12, 17, 24, 39, 59, 69)
age_labels <- c("Children", "Adolescents", "Young Adults", "Adults", "Middle-aged Adults", "Older Adults")
sleep$age_group <- cut(sleep$age, breaks = age_breaks, labels = age_labels, include.lowest = TRUE)

str(sleep)
summary(sleep)

# DATA EXPLORATION - KEY FINDING 1 ---------------------------------------------------------------------
# Key Finding 1 - Sleep Efficiency across Gender + Age Group
ggplot(sleep, aes(x = age_group, y = sleep_efficiency, fill = gender)) +
  geom_boxplot() +
  labs(title = "Sleep Efficiency Across Gender and Age Groups",
       x = "Age Group",
       y = "Sleep Efficiency",
       fill = "Gender") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

summaryBy(sleep_efficiency ~ age_group + gender, data = sleep, FUN = summary)

ggplot(sleep, aes(x = gender, y = sleep_efficiency, fill = gender)) +
  geom_boxplot() +
  labs(title = "Sleep Efficiency by Gender",
       x = "Gender",
       y = "Sleep Efficiency",
       fill = "Gender") +
  theme_minimal()

ggplot(sleep, aes(x = age_group, y = sleep_efficiency, fill = age_group)) +
  geom_boxplot() +
  labs(title = "Sleep Efficiency by Age Group",
       x = "Age Group",
       y = "Sleep Efficiency",
       fill = "Age Group") +
  theme_minimal()

# DATA EXPLORATION - KEY FINDING 2 ---------------------------------------------------------------------
# Key Finding 2
test <- sleep[, sapply(sleep, is.numeric)]
corrplot(cor(test), method = 'number', type = 'upper')
correlation_data <- as.data.frame()

# DATA EXPLORATION - KEY FINDING 3 ---------------------------------------------------------------------
# Key Finding 3
ggplot(sleep, aes(x = bedtime, y = sleep_efficiency, fill = bedtime)) +
  geom_boxplot() +
  labs(title = "Sleep Efficiency by Bedtime",
       x = "Bedtime",
       y = "Sleep Efficiency",
       fill = "Bedtime") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
summaryBy(sleep_efficiency ~ bedtime, data = sleep, FUN = summary)


# RESEARCH QUESTION 1 PREP ----
lm_model1 <- lm(sleep_efficiency ~ sleep_duration + bedtime + wakeup_time + caffeine_consumption + alcohol_consumption + smoking_status + exercise_frequency + daily_steps + age + gender, data = sleep)
summary(lm_model1)
stepwise_model <- step(lm_model1, direction = "both")
summary(stepwise_model)

# RF MODEL WITH TRAIN-TEST SPLIT ----
# Train-Test Split
set.seed(1)
sleep_q1 <- sleep[, c("bedtime", "sleep_efficiency",
                      "alcohol_consumption", "smoking_status", 
                      "exercise_frequency", "daily_steps", "age")]
train_q1 <- sample.split(sleep_q1$sleep_efficiency, SplitRatio = 0.7)
trainset_q1 <- subset(sleep_q1, train_q1 == T)
testset_q1 <- subset(sleep_q1, train_q1 == F)

# RF Model
m.RF1 <- randomForest(sleep_efficiency ~ ., data=trainset_q1, importance=T)
m.RF1
plot(m.RF1)
## Confirms error stablised before 500 trees.
m.RF1.yhat <- predict(m.RF1, newdata = testset_q1)
RMSE.test.RF1 <- sqrt(mean((testset_q1$sleep_efficiency - m.RF1.yhat)^2))
RMSE.test.RF1 
var.impt.RF <- importance(m.RF1)
varImpPlot(m.RF1, type = 1)
# alcohol_consumption = the most important
# %IncMSE - Mean Decrease Accuracy - This shows how much our model accuracy decreases if we leave out that variable.

# Q1 MODEL 2: MARS ---------------------------------------------------------------------------
set.seed(1)
# Train MARS model degree 1
m1.mars1 <- earth(sleep_efficiency ~ ., degree=1, data=trainset_q1)
m1.mars1.yhat <- predict(m1.mars1, newdata = testset_q1)
RMSE.test.mars1.q1 <- sqrt(mean((testset_q1$sleep_efficiency - m1.mars1.yhat)^2))
RMSE.test.mars1.q1

# Train MARS model degree 2
m1.mars2 <- earth(sleep_efficiency ~ ., degree=2, data=trainset_q1)
m1.mars2.yhat <- predict(m1.mars2, newdata = testset_q1)
RMSE.test.mars2.q1 <- sqrt(mean((testset_q1$sleep_efficiency - m1.mars2.yhat)^2))
RMSE.test.mars2.q1

var.impt.mars1 <- evimp(m1.mars1)
print(var.impt.mars1)

# RESEARCH QUESTION 2 PREP ----
lm_model2 <- lm(sleep_efficiency ~ rem_sleep_percentage + deep_sleep_percentage + light_sleep_percentage + awakenings, data = sleep)
summary(lm_model2)
stepwise_model2 <- step(lm_model2, direction = "both")
summary(stepwise_model2)

# Q2 MODEL 1: RF ------------------------
set.seed(1)
sleep_q2 <- sleep[, c("awakenings", "sleep_efficiency", "rem_sleep_percentage", "deep_sleep_percentage")]
train_q2 <- sample.split(sleep_q2$sleep_efficiency, SplitRatio = 0.7)
trainset_q2 <- subset(sleep_q2, train_q2 == T)
testset_q2 <- subset(sleep_q2, train_q2 == F)

# RF Model
m.RF2 <- randomForest(sleep_efficiency ~ ., data=trainset_q2, importance=T)
m.RF2
m.RF2.pred <- predict(m.RF2, newdata = testset_q2)
plot(m.RF2)
## Confirms error stablised before 500 trees.
m.RF2.yhat <- predict(m.RF2, newdata = testset_q2)
RMSE.test.RF2 <- sqrt(mean((testset_q2$sleep_efficiency - m.RF2.yhat)^2))
RMSE.test.RF2
var.impt.RF <- importance(m.RF2)
varImpPlot(m.RF2, type = 1)
# deep_sleep_percentage = the most important
# %IncMSE - Mean Decrease Accuracy - This shows how much our model accuracy decreases if we leave out that variable.

# Q2 MODEL 2: MARS -------
set.seed(1)
# Train MARS model degree 1
m2.mars1 <- earth(sleep_efficiency ~ ., degree=1, data=trainset_q2)
m2.mars1.yhat <- predict(m2.mars1, newdata = testset_q2)
RMSE.test.mars1.q2 <- sqrt(mean((testset_q1$sleep_efficiency - m2.mars1.yhat)^2))
RMSE.test.mars1.q2

# Train MARS model degree 2
m2.mars2 <- earth(sleep_efficiency ~ ., degree=2, data=trainset_q2)
m2.mars2.yhat <- predict(m2.mars2, newdata = testset_q2)
RMSE.test.mars2.q2 <- sqrt(mean((testset_q1$sleep_efficiency - m2.mars2.yhat)^2))
RMSE.test.mars2.q2

var.impt.mars2 <- evimp(m2.mars2)
print(var.impt.mars2)
