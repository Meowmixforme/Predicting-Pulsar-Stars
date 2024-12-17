library(caret)
library(e1071)
library(randomForest)
library(reticulate)
library(tidyr)
library(dplyr)
library(ggplot2)
library(ggthemes)
library(corrplot)
library(smotefamily)


# Set script current directory as working directory

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Read CSV file 

pulsar <- read.csv("HTRU_2.csv")


# EDA

str(pulsar)
class(pulsar)
print("Overall class distribution:")
table(pulsar$X0)
prop.table(table(pulsar$X0)) * 100 
# Dimensions of dataset

dim(pulsar)

# list types for each attribute

sapply(pulsar, class)

# Take a peek at the first 20 rows of the dataset

head(pulsar, n=20)

# summarize attribute distributions

summary(pulsar)


# Number of missing values

colSums(is.na(pulsar))

# Unimodal data visualizations

# histograms each attribute

par(mfrow = c(3,3))

for (i in 1:9) {
  hist(pulsar[, i], main = names(pulsar)[i],
       xlab = names(pulsar)[i], 
       col = ifelse(pulsar$X0 == 0, "lightblue", "lightcoral"),
       border = "black",
       breaks = 30) 
}


# density plot for each attribute

par(mfrow = c(3,3))
for (i in 1:9) {
  plot(density(pulsar[,i]), main = names(pulsar)[i])
  
}

# Boxplots for each attribute

par(mfrow = c(3, 3))


for (i in 1:8) {
  boxplot(pulsar[, i] ~ pulsar$X0, 
          main = names(pulsar)[i], 
          xlab = "Pulsar Type", 
          ylab = names(pulsar)[i],
          names = c("Non-Pulsar (0)", "Pulsar (1)"))
}




# scatterplot matrix

colors <- ifelse(pulsar$X0 == 0, "blue", "red") 
pairs(pulsar[,1:9],
      col = colors, 
      pch = 19,  # Use filled circles
      main = "Scatter Plot Matrix of Pulsar Predictors",
      upper.panel = NULL) 

# correlation plot

correlations <- cor(pulsar[,1:8])

corrplot(correlations, method = "circle")


# Preprocessing

# Split the data into x and y to split features from target
y <- pulsar$X0 
X <- subset(pulsar, select = -X0)

# Train-test split

# Set random seed and shuffle data
set.seed(123)
pulsar <- pulsar[sample(nrow(pulsar)), ]

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

# Apply BLSMOTE on training data
X_train_features <- X_train[, !names(X_train) %in% c("Class")]

# Convert Class labels to numeric (0,1) properly
y_train_blsmote <- ifelse(X_train$Class == "Class0", 0, 1)

# Verify no NAs in the data
print("Checking for NAs in features:")
print(colSums(is.na(X_train_features)))
print("\nChecking for NAs in target:")
print(sum(is.na(y_train_blsmote)))

# Apply BLSMOTE
genData_BLSMOTE <- BLSMOTE(X_train_features, 
                           y_train_blsmote, 
                           K = 5,  
                           C = 5)

# Combine original and synthetic data
combined_data <- rbind(genData_BLSMOTE$data, genData_BLSMOTE$syn_data)

# Convert class to factor with proper labels
combined_data$class <- factor(combined_data$class, 
                              levels = c(0, 1), 
                              labels = c("Class0", "Class1"))

# Now perform downsampling on the combined data
majority_class <- combined_data[combined_data$class == "Class0", ]
minority_class <- combined_data[combined_data$class == "Class1", ]

# Set target number for downsampling (using your target of 2500)
target_samples <- 2500

# Downsample majority class
set.seed(123)
majority_downsampled <- majority_class[sample(nrow(majority_class), 
                                              size = target_samples, 
                                              replace = FALSE), ]

# Take samples from minority class
minority_samples <- minority_class[sample(nrow(minority_class), 
                                          size = target_samples, 
                                          replace = FALSE), ]

# Combine datasets
X_train <- rbind(majority_downsampled, minority_samples)

# Shuffle the final dataset
X_train <- X_train[sample(nrow(X_train)), ]

# Rename the class column to match your original code
names(X_train)[names(X_train) == "class"] <- "Class"

# Verify the class distribution after balancing
print("New class distribution after BLSMOTE and downsampling:")
print(prop.table(table(X_train$Class)) * 100)

# Check the number of samples in each class
cat("Number of samples in Class0:", nrow(X_train[X_train$Class == "Class0", ]), "\n")
cat("Number of samples in Class1:", nrow(X_train[X_train$Class == "Class1", ]), "\n")

# Convert back to dataframes
train_data <- as.data.frame(X_train)
X_validation <- as.data.frame(X_validation)

# make sure y_validation is a factor with the same levels
y_validation <- factor(y_validation, levels = c(0, 1), labels = c("Class0", "Class1"))

# Prepare validation data
X_validation <- as.data.frame(X_validation)





### testing ###



# Set up cross-validation parameters
trainControl <- trainControl(method = "repeatedcv",number = 10,repeats = 3,classProbs = TRUE,summaryFunction = twoClassSummary,savePredictions = TRUE)


# Train model with cross-validation
glm_model <-  caret::train(Class ~ ., data = X_train, method = "glm",family = "binomial", trControl = trainControl,metric = "ROC")

summary (glm_model)

# Print results
print(glm_model)
print(glm_model$results)

# Make predictions on validation set
predictions <- predict(glm_model, newdata = as.data.frame(X_validation))
confusion_matrix <- confusionMatrix(predictions, y_validation)
print("Confusion Matrix and Statistics glm:")
print(confusion_matrix)


# glmnet
# Train model with cross-validation
glmnet_model <- train(Class ~ .,data = X_train,method ="glmnet",trControl = trainControl,metric = "ROC")

summary (glmnet_model)
plot(glmnet_model)

# Print results
print("Best tuning parameters:")
print(glmnet_model$bestTune)
print("\nModel performance across parameters:")
print(glmnet_model)

# Make predictions
predictions <- predict(glmnet_model, newdata = as.data.frame(X_validation))
confusion_matrix <- confusionMatrix(predictions, y_validation)
print("Confusion Matrix and Statistics glmnet:")
print(confusion_matrix)

# glmnet ridge tuned
# Set up tuning grid for glmnet ridge
tuneGrid_r <- expand.grid(alpha = seq(0, 1, by = 0.1),lambda = seq(0.01, 0.001, length = 10)) # Best settings

# Train the tuned glmnet model ridge
set.seed(123)  # for reproducibility
glmnet_tuned_r <- caret::train(Class ~ .,data = X_train,method = "glmnet",trControl = trainControl,tuneGrid = tuneGrid_r,metric = "ROC")

summary (glmnet_tuned_r)
plot(glmnet_tuned_r)

# Make predictions
predictions <- predict(glmnet_tuned_r, newdata = X_validation)
prob_predictions <- predict(glmnet_tuned_r, newdata = X_validation, type = "prob")

# Create confusion matrix
confusion_matrix <- confusionMatrix(predictions, y_validation)
print("Confusion Matrix and Statistics ridge:")
print(confusion_matrix)

# # glmnet lasso tuned
# Set up tuning grid for glmnet lasso
tuneGrid_l <- expand.grid(alpha = seq(1, 1, by = 0.2),lambda = seq(0.1, 0.01, length = 10)) # Best settings

# Train the tuned glmnet model lasso
set.seed(123)  # for reproducibility
glmnet_tuned_l <- caret::train(Class ~ .,data = X_train,method = "glmnet",trControl = trainControl,tuneGrid = tuneGrid_l,metric = "ROC")

summary (glmnet_tuned_l)
plot(glmnet_tuned_l)

# Make predictions
predictions <- predict(glmnet_tuned_l, newdata = X_validation)
prob_predictions <- predict(glmnet_tuned_l, newdata = X_validation, type = "prob")

# Create confusion matrix
confusion_matrix <- confusionMatrix(predictions, y_validation)
print("Confusion Matrix and Statistics lasso:")
print(confusion_matrix)

# glmnet elastic net tuned
# Set up tuning grid for glmnet elastic
tuneGrid_e <- expand.grid(alpha = seq(0.5, 1, by = 0.1),lambda = seq(0.1, 0.001, length = 10)) # Best settings

# Train the tuned glmnet model elastic net
set.seed(123)  # for reproducibility
glmnet_tuned_e <- caret::train(Class ~ .,data = X_train,method = "glmnet",trControl = trainControl,tuneGrid = tuneGrid_e,metric = "ROC")

summary (glmnet_tuned_e)

# Make predictions
predictions <- predict(glmnet_tuned_e, newdata = X_validation)
prob_predictions <- predict(glmnet_tuned_e, newdata = X_validation, type = "prob")

# Create confusion matrix
confusion_matrix <- confusionMatrix(predictions, y_validation)
print("Confusion Matrix and Statistics Elastic net:")
print(confusion_matrix)

plot(glmnet_tuned_e)

# K-NN

# KNN-specific cross-validation
trainControlknn <- trainControl(method = "repeatedcv",number = 10,repeats = 3,classProbs = TRUE,summaryFunction = twoClassSummary)

# KNN
set.seed(123)  # for reproducibility
knn_model <- caret::train(  Class ~ .,data = X_train,method = "knn",trControl = trainControl,metric = "ROC")

summary (knn_model)

# Make predictions
predictions <- predict(knn_model, newdata = X_validation)
prob_predictions <- predict(knn_model, newdata = X_validation, type = "prob")

# Create confusion matrix
confusion_matrix <- confusionMatrix(predictions, y_validation)
print("Confusion Matrix and Statistics K-NN:")
print(confusion_matrix)

# K-NN tuned


# Create tuning grid for KNN
tuneGrid_knn <- expand.grid(k = seq(7, 9, by = 1)) # 8 is peak

set.seed(123)  # for reproducibility
knn_model_t <- caret::train(Class ~ .,data = X_train,method = "knn",trControl = trainControlknn,tuneGrid = tuneGrid_knn,metric = "ROC") # was worse with PCA

summary (knn_model_t) 
plot(knn_model_t)

# Make predictions
predictions <- predict(knn_model_t, newdata = X_validation)
prob_predictions <- predict(knn_model_t, newdata = X_validation, type = "prob")

# Create confusion matrix
confusion_matrix <- confusionMatrix(predictions, y_validation)
print("Confusion Matrix and Statistics K-NN tuned:")
print(confusion_matrix)



# SVM radial Tuned
tuneGrid_svm_r <- expand.grid(sigma = seq(0.01, 2, length = 10), C = seq(0.01, 5, length = 10))

# Set seed for reproducibility
set.seed(123)

# Use the tune function for hyperparameter tuning
tuned_svm_r <- tune(svm, Class ~ ., data = X_train, kernel = "radial", ranges = tuneGrid_svm_r, tunecontrol = tune.control(cross = 10))

# Print the best parameters
plot(tuned_svm_r)
summary(tuned_svm_r)

# Train the final model with the best parameters
final_model_r <- tuned_svm_r$best.model

# Make predictions with the final model on the validation set
predictions_tuned <- predict(final_model_r, newdata = X_validation)

# Create confusion matrix for the final model
confusion_matrix_tuned <- confusionMatrix(predictions_tuned, y_validation)
print("Confusion Matrix and Statistics SVM Tuned:")
print(confusion_matrix_tuned)


# SVM polynomial Tuned
tuneGrid_svm_poly <- expand.grid(degree = 2:3, C = seq(0.1, 2, length = 5), scale = seq(0.1, 1, length = 5))

# Set seed for reproducibility
set.seed(123)

# Use the tune function for hyperparameter tuning with polynomial kernel
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
print("Confusion Matrix and Statistics SVM Tuned:")
print(confusion_matrix_tuned)


# SVM Linear Tuned
tuneGrid_svm_linear <- expand.grid(C = seq(0.1, 2, length = 5))

# Set seed for reproducibility
set.seed(123)

# Use the tune function for hyperparameter tuning with linear kernel
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
print("Confusion Matrix and Statistics for Linear SVM Tuned:")
print(confusion_matrix_tuned)

# SVM Sigmoid Tuned
tuneGrid_svm_sigmoid <- expand.grid(C = seq(0.1, 2, length = 5), alpha = seq(0.1, 1, length = 5))

# Set seed for reproducibility
set.seed(123)

# Use the tune function for hyperparameter tuning with sigmoid kernel
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
print("Confusion Matrix and Statistics SVM Tuned:")
print(confusion_matrix_tuned)


## RANDOM FOREST ##


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

# Random Forest Tuned

# Set up cross-validation parameters
trainControl_rf <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary)

# Create a tuning grid for mtry
tuneGrid_rf <- expand.grid(mtry = seq(2, ncol(X_train) - 1, by = 1)) # Adjust the range as needed

# Train the Random Forest model with tuning
set.seed(123)  # for reproducibility
rf_model_tuned <- caret::train(Class ~ ., data = X_train, method = "rf",trControl = trainControl_rf,tuneGrid = tuneGrid_rf,metric = "ROC",ntree = 70)  # Set ntree directly here

# Print the model summary
print(rf_model_tuned)

# Plot the tuning results
plot(rf_model_tuned)

# Make predictions on the validation set
rf_predictions_tuned <- predict(rf_model_tuned, newdata = X_validation)

# Create confusion matrix for the tuned Random Forest model
rf_confusion_matrix_tuned <- confusionMatrix(rf_predictions_tuned, y_validation)
print("Confusion Matrix and Statistics for Tuned Random Forest:")
print(rf_confusion_matrix_tuned)


