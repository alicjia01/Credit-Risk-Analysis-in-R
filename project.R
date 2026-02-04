# Clear environment
rm(list = ls())

# Load core packages for data manipulation and visualization
library(dplyr)
library(ggplot2)

# Load packages for statistical and machine learning models
library(rpart)
library(rpart.plot)
library(randomForest)

# Load packages for model evaluation
library(ROCR)
library(caret)

# Ensure reproducibility of results
set.seed(123)

# Load the credit dataset
data <- read.csv("cs-training.csv")

# Remove the ID column
data <- data[, -1]

# Convert target variable to a factor
data$SeriousDlqin2yrs <- as.factor(data$SeriousDlqin2yrs)

# Impute missing values using median imputation
data <- data %>%
  mutate(across(where(is.numeric),
                ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))

# Function to cap extreme values at selected quantiles
cap_outliers <- function(x, lower = 0.01, upper = 0.99) {
  qnt <- quantile(x, probs = c(lower, upper))
  x[x < qnt[1]] <- qnt[1]
  x[x > qnt[2]] <- qnt[2]
  return(x)
}

# Apply outlier capping to all numeric variables
data <- data %>%
  mutate(across(where(is.numeric), cap_outliers))

# Split data into training and test sets (70/30)
n <- nrow(data)
train_idx <- sample(1:n, size = 0.7 * n)

train <- data[train_idx, ]
test  <- data[-train_idx, ]

# Benchmark model: constant probability equal to training default rate
p_benchmark <- mean(as.numeric(as.character(train$SeriousDlqin2yrs)))
benchmark_pred <- rep(p_benchmark, nrow(test))

# Logistic regression model
logit_model <- glm(
  SeriousDlqin2yrs ~ .,
  data = train,
  family = binomial()
)

# Predicted probabilities from logistic regression
logit_prob <- predict(logit_model, test, type = "response")

# Decision tree model with controlled complexity
tree_model <- rpart(
  SeriousDlqin2yrs ~ .,
  data = train,
  method = "class",
  control = rpart.control(
    cp = 0.005,
    maxdepth = 4,
    minsplit = 500
  )
)

# Save decision tree visualization
png("tree_model.png", width = 900, height = 600)
rpart.plot(tree_model, type = 2, extra = 104)
dev.off()

# Predicted probabilities from decision tree
tree_prob <- predict(tree_model, test)[, 2]

# Random forest model with class weights to address class imbalance
rf_model <- randomForest(
  SeriousDlqin2yrs ~ .,
  data = train,
  ntree = 500,
  mtry = floor(sqrt(ncol(train) - 1)),
  importance = TRUE,
  classwt = c("0" = 1, "1" = 4)
)

# Predicted probabilities from random forest
rf_prob <- predict(rf_model, test, type = "prob")[, 2]

# Save random forest variable importance plot
png("rf_variable_importance.png", width = 900, height = 600)
par(mar = c(5, 14, 4, 2))
varImpPlot(rf_model, n.var = 8, main = "Top Variable Importance (Random Forest)")
dev.off()

# Function to compute AUC
get_auc <- function(true, prob) {
  pred <- prediction(prob, true)
  perf <- performance(pred, "auc")
  as.numeric(perf@y.values)
}

# Compare model performance using AUC
auc_results <- data.frame(
  Model = c("Benchmark", "Logit", "Decision Tree", "Random Forest"),
  AUC = c(
    get_auc(test$SeriousDlqin2yrs, benchmark_pred),
    get_auc(test$SeriousDlqin2yrs, logit_prob),
    get_auc(test$SeriousDlqin2yrs, tree_prob),
    get_auc(test$SeriousDlqin2yrs, rf_prob)
  )
)

print(auc_results)

# Save ROC curves for all models
png("roc_curves.png", width = 900, height = 600)

plot(
  performance(prediction(benchmark_pred, test$SeriousDlqin2yrs), "tpr", "fpr"),
  col = "black", lwd = 2
)

plot(performance(prediction(logit_prob, test$SeriousDlqin2yrs), "tpr", "fpr"),
     col = "blue", lwd = 2, add = TRUE)

plot(performance(prediction(tree_prob, test$SeriousDlqin2yrs), "tpr", "fpr"),
     col = "red", lwd = 2, add = TRUE)

plot(performance(prediction(rf_prob, test$SeriousDlqin2yrs), "tpr", "fpr"),
     col = "darkgreen", lwd = 2, add = TRUE)

legend(
  "bottomright",
  legend = c("Benchmark", "Logit", "Tree", "Random Forest"),
  col = c("black", "blue", "red", "darkgreen"),
  lwd = 2
)

dev.off()

# Classification threshold adjusted for class imbalance
threshold <- 0.3

rf_class <- factor(ifelse(rf_prob > threshold, 1, 0))
logit_class <- factor(ifelse(logit_prob > threshold, 1, 0))

# Confusion matrices for logistic regression and random forest
cm_rf <- confusionMatrix(rf_class, test$SeriousDlqin2yrs, positive = "1")
cm_logit <- confusionMatrix(logit_class, test$SeriousDlqin2yrs, positive = "1")

cm_rf
cm_logit

# Save confusion matrices to text files
capture.output(cm_rf, file = "confusion_matrix_rf.txt")
capture.output(cm_logit, file = "confusion_matrix_logit.txt")
