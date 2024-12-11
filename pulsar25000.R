library(caret)
library(e1071)
library(randomForest)
library(keras)
library(tensorflow)
library(reticulate)
library(tidyr)
library(ggplot2)
library(ggthemes)
library(corrplot)


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

par(mfrow = c(2,7))

for (i in 1:13) {
  hist(pulsar[, i], main = names(pulsar)[i])
  
}


# density plot for each attribute

par(mfrow = c(2,7))
for (i in 1:9) {
  plot(density(pulsar[,i]), main = names(pulsar)[i])
  
}

# Boxplots for each attribute

par(mfrow = c(2,7))
for (i in 1:9) {
  boxplot(pulsar[,i], main = names(pulsar)[i])
  
}

# scatterplot matrix

pairs(pulsar[,1:9])

# correlation plot

correlations <- cor(pulsar[,1:8])

corrplot(correlations, method = "circle")


# Create a long version of the dataset
pulsar2 <- gather(pulsar, "feature", "value", -X0)  # Exclude the target variable X0

# Create the boxplot
ggplot(pulsar2) +
  geom_boxplot(aes(factor(X0), log(value))) +
  facet_wrap(~feature, scales = "free") +
  labs(title = "Box-plot of all predictors(log scaled) per pulsar type",
       subtitle = "Pulsar can be either non-pulsar (0) or pulsar (1)") +
  theme_fivethirtyeight() +
  theme(axis.title = element_text()) +
  ylab("Predictor's log value") +
  xlab('')


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

# Separate majority and minority classes
majority_class <- X_train[X_train$Class == "Class0", ]
minority_class <- X_train[X_train$Class == "Class1", ]

# Set target number of samples for each class
target_samples <- 2500  # 2500 for each class

# Random downsampling of majority class to reach the target number
set.seed(123)
if (nrow(majority_class) > target_samples) {
  majority_downsampled <- majority_class[sample(nrow(majority_class), 
                                                size = target_samples, 
                                                replace = FALSE), ]
} else {
  majority_downsampled <- majority_class  # No downsampling needed if already sufficient
}

# Random upsampling of minority class to reach the target number
if (nrow(minority_class) < target_samples) {
  minority_upsampled <- minority_class[sample(nrow(minority_class), 
                                              size = target_samples, 
                                              replace = TRUE), ]
} else {
  minority_upsampled <- minority_class  # No upsampling needed if already sufficient
}

# Combine downsampled majority class with upsampled minority class
X_train <- rbind(majority_downsampled, minority_upsampled)

# Shuffle the balanced dataset
X_train <- X_train[sample(nrow(X_train)), ]

# Verify the class distribution after balancing
print("New class distribution after balancing:")
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


# feature selection made no difference to scores



# Training

# Set up cross-validation parameters
trainControl <- trainControl(method = "repeatedcv",number = 10,repeats = 3,classProbs = TRUE,summaryFunction = twoClassSummary,savePredictions = TRUE)


# Train model with cross-validation
glm_model <- train( Class ~ .,data = X_train,method = "glm",family = "binomial",trControl = trainControl,metric = "Accuracy")

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
tuneGrid_r <- expand.grid(alpha = seq(0, 1, by = 0.4),lambda = seq(0.01, 0.001, length = 10)) # Best settings

# Train the tuned glmnet model ridge
set.seed(123)  # for reproducibility
glmnet_tuned_r <- train(Class ~ .,data = X_train,method = "glmnet",trControl = trainControl,tuneGrid = tuneGrid_r,metric = "ROC")

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
tuneGrid_l <- expand.grid(alpha = seq(1, 1, by = 0.1),lambda = seq(0.1, 0.01, length = 10)) # Best settings

# Train the tuned glmnet model lasso
set.seed(123)  # for reproducibility
glmnet_tuned_l <- train(Class ~ .,data = X_train,method = "glmnet",trControl = trainControl,tuneGrid = tuneGrid_l,metric = "ROC")

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
tuneGrid_e <- expand.grid(alpha = seq(0.5, 1, by = 0.3),lambda = seq(0.1, 0.001, length = 10)) # Best settings

# Train the tuned glmnet model elastic net
set.seed(123)  # for reproducibility
glmnet_tuned_e <- train(Class ~ .,data = X_train,method = "glmnet",trControl = trainControl,tuneGrid = tuneGrid_e,metric = "ROC")

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
knn_model <- train(  Class ~ .,data = X_train,method = "knn",trControl = trainControl,metric = "ROC")

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
tuneGrid_knn <- expand.grid(k = seq(7, 9, by = 1))

set.seed(123)  # for reproducibility
knn_model_t <- train(Class ~ .,data = X_train,method = "knn",trControl = trainControlknn,tuneGrid = tuneGrid_knn,metric = "ROC") # was worse with PCA

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
tuneGrid_rf <- expand.grid(mtry = seq(1, ncol(X_train) - 1, by = 1))  # Adjust the range as needed

# Train the Random Forest model with tuning
set.seed(123)  # for reproducibility
rf_model_tuned <- train(Class ~ ., data = X_train, method = "rf",trControl = trainControl_rf,tuneGrid = tuneGrid_rf,metric = "ROC",ntree = 70)  # Set ntree directly here

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


# Keras 

# Convert class labels to categorical
y_train_categorical <- to_categorical(as.numeric(y_train) - 1)  # Assuming y_train is a factor
y_validation_categorical <- to_categorical(as.numeric(y_validation) - 1)

# Define the CNN model
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu', input_shape = c(28, 28, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 2, activation = 'softmax')  # Adjust units for the number of classes

# Compile the model
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)

# Train the model
history <- model %>% fit(
  X_train, y_train_categorical,
  epochs = 30,  # Adjust the number of epochs as needed
  batch_size = 32,  # Adjust batch size as needed
  validation_data = list(X_validation, y_validation_categorical)
)

# Evaluate the model on the validation set
score <- model %>% evaluate(X_validation, y_validation_categorical)
cat("Validation loss:", score$loss, "\n")
cat("Validation accuracy:", score$accuracy, "\n")

# Make predictions on the validation set
predictions <- model %>% predict_classes(X_validation)

# Create confusion matrix for the CNN model
library(caret)
confusion_matrix_cnn <- confusionMatrix(as.factor(predictions), as.factor(y_validation))
print("Confusion Matrix and Statistics for CNN:")
print(confusion_matrix_cnn)