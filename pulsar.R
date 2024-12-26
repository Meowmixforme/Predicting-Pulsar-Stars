# v8255920 James Fothergill


library(caret)
library(e1071)
library(rpart)
library(randomForest)
library(nnet)
library(tidyr)
library(dplyr)
library(ggplot2)
library(ggthemes)
library(corrplot)


# Set the current directory as the working directory

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Read the CSV file 

pulsar <- read.csv("HTRU_2.csv")

## EDA

# view the dataset structure

str(pulsar)

sapply(pulsar, class)

# view the dataset dimensions (rows and columns)

dim(pulsar)


# The overall class distribution

table(pulsar$X0)

# The class distrobution as a percentage

prop.table(table(pulsar$X0)) * 100 


# Number of missing values

colSums(is.na(pulsar))

# Take a peek at the first 20 rows of the dataset

head(pulsar, n=20)

# summarize attribute distributions

summary(pulsar)


## Data visualizations

# histograms each attribute

par(mfrow = c(3,3))

for (i in 1:9) {
  hist(pulsar[, i], main = names(pulsar)[i],
       xlab = names(pulsar)[i], 
       col = ifelse(pulsar$X0 == 0, "lightblue", "lightcoral"),
       border = "black",
       breaks = 30) 
}

# clearing previous plot as R has errors

dev.off() 


# density plot for each attribute

par(mfrow = c(3,3))
for (i in 1:9) {
  plot(density(pulsar[,i]), main = names(pulsar)[i])
  
}

dev.off()

# Boxplots for each attribute

par(mfrow = c(3, 3))


for (i in 1:8) {
  boxplot(pulsar[, i] ~ pulsar$X0, 
          main = names(pulsar)[i], 
          xlab = "Pulsar Type", 
          ylab = names(pulsar)[i],
          names = c("Non-Pulsar (0)", "Pulsar (1)"))
}


dev.off() 


# scatterplot matrix

colors <- ifelse(pulsar$X0 == 0, "blue", "red") 
pairs(pulsar[,1:9],
      col = colors, 
      pch = 19,  # Use filled circles
      main = "Scatter Plot Matrix of Pulsar Predictors",
      upper.panel = NULL) 

dev.off()


# correlation plot

correlations <- cor(pulsar[,1:8])

corrplot(correlations, method = "circle")

dev.off()

# Preprocessing

# Set random seed and shuffle data

set.seed(123)
pulsar <- pulsar[sample(nrow(pulsar)), ]

# Train-test split

# Create split indices

validationIndex <- createDataPartition(pulsar$X0, p = .70, list = FALSE)

# Create training and validation sets

testing <- pulsar[validationIndex, ]
validation <- pulsar[-validationIndex, ]

# Separate features and target for testing set

X_train <- testing[, !names(testing) %in% c("X0")]
y_train <- testing$X0

# Separate features and target for validation set

X_validation <- validation[, !names(validation) %in% c("X0")]
y_validation <- validation$X0

# Scale X_train

X_train <- scale(X_train)
X_validation <- scale(X_validation, center = attr(X_train, "scaled:center"),
                      scale = attr(X_train, "scaled:scale"))

# Convert to data frame and add class

X_train <- data.frame(X_train)
X_train$Class <- factor(y_train, levels = c(0, 1), labels = c("Class0", "Class1"))

# Separate majority and minority classes

majority_class <- X_train[X_train$Class == "Class0", ]
minority_class <- X_train[X_train$Class == "Class1", ]

# Random upsampling of minority class

set.seed(123)
minority_upsampled <- minority_class[sample(nrow(minority_class), 
                                            size = nrow(majority_class), 
                                            replace = TRUE), ]

# Combine majority class with upsampled minority class

X_train <- rbind(majority_class, minority_upsampled)

# Shuffle the balanced dataset

X_train <- X_train[sample(nrow(X_train)), ]

# Verify the new class distribution of X_train and natural distribution of y_train

table(X_train$Class)
prop.table(table(X_train$Class)) * 100
prop.table(table(y_train)) * 100

# Convert back to dataframes

train_data <- as.data.frame(X_train)
X_validation <- as.data.frame(X_validation)

# Ensure y_validation is a factor with the same levels

y_validation <- factor(y_validation, levels = c(0, 1), labels = c("Class0", "Class1"))





# Training

## GLM

# cross-validation parameters


trainControl <- trainControl(method = "repeatedcv",number = 10,repeats = 3,classProbs = TRUE,summaryFunction = twoClassSummary,savePredictions = TRUE)

# Train model with cross-validation

glm_model <- train( Class ~ .,data = X_train,method = "glm",family = "binomial",trControl = trainControl,metric = "ROC")



# Print results

summary (glm_model)
print(glm_model)
print(glm_model$results)

# Make predictions on validation set

predictions <- predict(glm_model, newdata = as.data.frame(X_validation))
confusion_matrix <- confusionMatrix(predictions, y_validation)
print("Confusion Matrix and Statistics glm: ")
print(confusion_matrix)


## glmnet

# Train model with cross-validation

glmnet_model <- train(Class ~ .,data = X_train,method ="glmnet",trControl = trainControl,metric = "ROC")

# Print results

summary (glmnet_model)
plot(glmnet_model)

# Print results
print("Best tuning parameters:")
print(glmnet_model$bestTune)
print(glmnet_model)

# Make predictions

predictions <- predict(glmnet_model, newdata = as.data.frame(X_validation))
confusion_matrix <- confusionMatrix(predictions, y_validation)
print("Confusion Matrix and Statistics glmnet:")
print(confusion_matrix)



## glmnet ridge tuned

# Set up tuning grid for glmnet ridge

tuneGrid_r <- expand.grid(alpha = 0, lambda = seq(0.01, 0.001, length = 10)) # Best settings

# Train the tuned glmnet model ridge

set.seed(123)  # for reproducibility
glmnet_tuned_r <- train(Class ~ .,data = X_train,method = "glmnet",trControl = trainControl,tuneGrid = tuneGrid_r,metric = "ROC")

# Print results

summary (glmnet_tuned_r)
plot(glmnet_tuned_r)
print(glmnet_tuned_r$bestTune)

# Make predictions

predictions <- predict(glmnet_tuned_r, newdata = X_validation)

# Create confusion matrix

confusion_matrix <- confusionMatrix(predictions, y_validation)
print("Confusion Matrix and Statistics ridge:")
print(confusion_matrix)



## glmnet lasso tuned

# Set up tuning grid for glmnet lasso

tuneGrid_l <- expand.grid(alpha = 1 ,lambda = 0.01) # Best settings

# Train the tuned glmnet model lasso

set.seed(123)  # for reproducibility
glmnet_tuned_l <- train(Class ~ .,data = X_train,method = "glmnet",trControl = trainControl,tuneGrid = tuneGrid_l,metric = "ROC")

summary (glmnet_tuned_l)

# Make predictions

predictions <- predict(glmnet_tuned_l, newdata = X_validation)

# Create confusion matrix

confusion_matrix <- confusionMatrix(predictions, y_validation)
print("Confusion Matrix and Statistics lasso:")
print(confusion_matrix)

# glmnet elastic net

# Set up tuning grid for glmnet elastic

tuneGrid_e <- expand.grid(alpha = seq(0.5, 0.8, by = 0.3),lambda = seq(0.1, 0.001, length = 10)) # Best settings

# Train the tuned glmnet model elastic net
set.seed(123)  # for reproducibility
glmnet_tuned_e <- train(Class ~ .,data = X_train,method = "glmnet",trControl = trainControl,tuneGrid = tuneGrid_e,metric = "ROC")

summary (glmnet_tuned_e)
plot(glmnet_tuned_e)
print(glmnet_tuned_e$bestTune)

# Make predictions
predictions <- predict(glmnet_tuned_e, newdata = X_validation)

# Create confusion matrix
confusion_matrix <- confusionMatrix(predictions, y_validation)
print("Confusion Matrix and Statistics Elastic net:")
print(confusion_matrix)



# K-NN

# KNN-specific cross-validation

trainControlknn <- trainControl(method = "repeatedcv",number = 10,repeats = 3,classProbs = TRUE,summaryFunction = twoClassSummary)

# KNN
set.seed(123)  # for reproducibility

knn_model <- train(  Class ~ .,data = X_train,method = "knn",trControl = trainControl,metric = "ROC")

summary (knn_model)

# Make predictions

predictions <- predict(knn_model, newdata = X_validation)

# Create confusion matrix

confusion_matrix <- confusionMatrix(predictions, y_validation)
print("Confusion Matrix and Statistics K-NN:")
print(confusion_matrix)

# K-NN tuned


# Create tuning grid for KNN

tuneGrid_knn <- expand.grid(k = seq(1, 30, by = 1))

set.seed(123)  # for reproducibility

knn_model_t <- train(Class ~ .,data = X_train,method = "knn",trControl = trainControlknn,tuneGrid = tuneGrid_knn,metric = "ROC") # was worse with PCA

summary (knn_model_t) 
plot(knn_model_t)
print(knn_model_t$bestTune)

# Make predictions

predictions <- predict(knn_model_t, newdata = X_validation)

# Create confusion matrix

confusion_matrix <- confusionMatrix(predictions, y_validation)
print("Confusion Matrix and Statistics K-NN tuned:")
print(confusion_matrix)

## K-NN best tuned = 13


# Create tuning grid for KNN

tuneGrid_knn <- expand.grid(k = 13)

set.seed(123)  # for reproducibility
knn_model_t <- train(Class ~ .,data = X_train,method = "knn",trControl = trainControlknn,tuneGrid = tuneGrid_knn,metric = "ROC") # was worse with PCA

summary (knn_model_t) 

# Make predictions

predictions <- predict(knn_model_t, newdata = X_validation)

# Create confusion matrix

confusion_matrix <- confusionMatrix(predictions, y_validation)
print("Confusion Matrix and Statistics K-NN tuned 13:")
print(confusion_matrix)

# K-NN  tuned lower range

tuneGrid_knn <- expand.grid(k = seq(7, 9, by = 1))

set.seed(123)  # for reproducibility
knn_model_t <- train(Class ~ .,data = X_train,method = "knn",trControl = trainControlknn,tuneGrid = tuneGrid_knn,metric = "ROC") # was worse with PCA

summary (knn_model_t) 
plot(knn_model_t)
print(knn_model_t$bestTune)

# Make predictions

predictions <- predict(knn_model_t, newdata = X_validation)

# Create confusion matrix

confusion_matrix <- confusionMatrix(predictions, y_validation)
print("Confusion Matrix and Statistics K-NN tuned 2:")
print(confusion_matrix)



# K-NN best tuned = 8


tuneGrid_knn <- expand.grid(k = 8)

set.seed(123)  # for reproducibility
knn_model_t <- train(Class ~ .,data = X_train,method = "knn",trControl = trainControlknn,tuneGrid = tuneGrid_knn,metric = "ROC") # was worse with PCA

summary (knn_model_t) 

# Make predictions

predictions <- predict(knn_model_t, newdata = X_validation)

# Create confusion matrix

confusion_matrix <- confusionMatrix(predictions, y_validation)
print("Confusion Matrix and Statistics K-NN tuned 8:")
print(confusion_matrix)




## SVM radial Tuned

tuneGrid_svm_r <- expand.grid(sigma = seq(0.01, 2, length = 5), C = seq(0.01, 5, length = 5))

# Tune the model

set.seed(123) # for reproducibility
tuned_svm_r <- tune(svm, Class ~ ., data = X_train, kernel = "radial", ranges = tuneGrid_svm_r, tunecontrol = tune.control(cross = 10))

# Print the best parameters

plot(tuned_svm_r)
summary(tuned_svm_r)
print(tuned_svm_r$bestTune)

# Train the final model with the best parameters

final_model_r <- tuned_svm_r$best.model

# Make predictions with the final model on the validation set

predictions_tuned <- predict(final_model_r, newdata = X_validation)

# Create confusion matrix for the final model

confusion_matrix_tuned <- confusionMatrix(predictions_tuned, y_validation)
print("Confusion Matrix and Statistics SVM radial Tuned:")
print(confusion_matrix_tuned)




## SVM polynomial Tuned

tuneGrid_svm_poly <- expand.grid(   degree = 2:4, C = seq(0.01, 5, length = 5), scale = seq(0.1, 1, length = 5))

# tune the model

set.seed(123) # for reproducibility
tuned_svm_poly <- tune(svm, Class ~ ., data = X_train, kernel = "polynomial", ranges = tuneGrid_svm_poly, tunecontrol = tune.control(cross = 10))

# Print the best parameters

plot(tuned_svm_poly)
summary(tuned_svm_poly)

# Train the final model with the best parameters

final_model_poly <- tuned_svm_poly$best.model

# Make predictions with the final model on the validation set

predictions_tuned <- predict(final_model_poly, newdata = X_validation)

# Create confusion matrix for the final model

confusion_matrix_tuned <- confusionMatrix(predictions_tuned, y_validation)
print("Confusion Matrix and Statistics SVM poly Tuned:")
print(confusion_matrix_tuned)




## SVM Linear Tuned

tuneGrid_svm_linear <- expand.grid(C = seq(0.1, 2, length = 5))

# tune model

set.seed(123) # for reproducibility
tuned_svm_linear <- tune(svm, Class ~ ., data = X_train, kernel = "linear", ranges = tuneGrid_svm_linear, tunecontrol = tune.control(cross = 10))

# Print the best parameters

plot(tuned_svm_linear)
summary(tuned_svm_linear)

# Train the final model with the best parameters

final_model_linear <- tuned_svm_linear$best.model

# Make predictions with the final model on the validation set

predictions_tuned <- predict(final_model_linear, newdata = X_validation)

# Create confusion matrix for the final model

confusion_matrix_tuned <- confusionMatrix(predictions_tuned, y_validation)
print("Confusion Matrix and Statistics for Linear SVM linear Tuned:")
print(confusion_matrix_tuned)

# SVM Sigmoid Tuned

tuneGrid_svm_sigmoid <- expand.grid(gamma = seq(0.001, 1, length = 5),coef0 = seq(0, 2, length = 5),C = seq(0.1, 5, length = 5))


# Use the tune function for hyperparameter tuning with sigmoid kernel

set.seed(123) # for reproducibility
tuned_svm_sigmoid <- tune(svm, Class ~ ., data = X_train, kernel = "sigmoid", ranges = tuneGrid_svm_sigmoid, tunecontrol = tune.control(cross = 10))

# Print the best parameters
plot(tuned_svm_sigmoid)
summary(tuned_svm_sigmoid)

# Train the final model with the best parameters

final_model_sigmoid <- tuned_svm_sigmoid$best.model

# Make predictions with the final model on the validation set

predictions_tuned <- predict(final_model_sigmoid, newdata = X_validation)

# Create confusion matrix for the final model

confusion_matrix_tuned <- confusionMatrix(predictions_tuned, y_validation)
print("Confusion Matrix and Statistics SVM sigmoid Tuned:")
print(confusion_matrix_tuned)



## RANDOM FOREST


# Train the Random Forest model with default parameters

set.seed(123)  # for reproducibility
rf_model <- randomForest(Class ~ ., data = X_train)

# Print the model summary

print(rf_model)
plot(rf_model)

# Make predictions on the validation set

rf_predictions <- predict(rf_model, newdata = X_validation)

# Create confusion matrix for the Random Forest model

rf_confusion_matrix <- confusionMatrix(rf_predictions, y_validation)
print("Confusion Matrix and Statistics for Random Forest:")
print(rf_confusion_matrix)

## Random Forest Tuned

# Set up cross-validation parameters

trainControl_rf <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary)

# Create a tuning grid for mtry

tuneGrid_rf <- expand.grid(mtry = seq(1, ncol(X_train) - 1, by = 1))

# Train the Random Forest model with tuning

set.seed(123)  # for reproducibility
rf_model_tuned <- caret::train(Class ~ ., data = X_train, method = "rf",trControl = trainControl_rf,tuneGrid = tuneGrid_rf,metric = "ROC",ntree = 80,nodesize = 2,maxnodes = 1000)  # Set ntree directly here

# Print the model summary

print(rf_model_tuned)

# Plot the tuning results

plot(rf_model_tuned)

# Make predictions on the validation set

rf_predictions_tuned <- predict(rf_model_tuned, newdata = X_validation)

# Create confusion matrix for the tuned Random Forest model

rf_confusion_matrix_tuned <- confusionMatrix(rf_predictions_tuned, y_validation)
print("Confusion Matrix and Statistics for Tuned Random Forest tuned:")
print(rf_confusion_matrix_tuned)

best_mtry <- rf_model_tuned$bestTune$mtry 
print(paste("Best mtry value:", best_mtry))

### Random forest best tuned

# Train Control

trainControl_rf <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary)

tuneGrid_rf <- expand.grid(mtry = 2)


# Train the Random Forest model with the best mtry value

set.seed(123)  # for reproducibility
rf_model_optimized <- caret::train(Class ~ ., data = X_train, method = "rf", trControl = trainControl_rf,tuneGrid = tuneGrid_rf,metric = "ROC",ntree = 80,nodesize = 2,maxnodes = 1000) 

# Print the model summary

print(rf_model_optimized)

# Make predictions

rf_predictions_optimized <- predict(rf_model_optimized, newdata = X_validation)

# Create confusion matrix optimized Random Forest

rf_confusion_matrix_optimized <- confusionMatrix(rf_predictions_optimized, y_validation)
print("Confusion Matrix and Statistics for Optimized Random Forest best:")
print(rf_confusion_matrix_optimized)


