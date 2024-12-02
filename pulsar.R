library(caret)



# Set script current directory as working directory

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Read CSV file 

pulsar <- read.csv("HTRU_2.csv")

# 

str(pulsar)
class(pulsar)
print("Overall class distribution:")
table(pulsar$X0)
prop.table(table(pulsar$X0)) * 100 


# Number of missing values

colSums(is.na(pulsar))


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

# Random upsampling of minority class
set.seed(123)
minority_upsampled <- minority_class[sample(nrow(minority_class), 
                                            size = nrow(majority_class), 
                                            replace = TRUE), ]

# Combine majority class with upsampled minority class
X_train <- rbind(majority_class, minority_upsampled)

# Shuffle the balanced dataset
X_train <- X_train[sample(nrow(X_train)), ]

# Verify the class distribution
print("Original class distribution:")
prop.table(table(y_train)) * 100

# Convert back to dataframes
train_data <- as.data.frame(X_train)
X_validation <- as.data.frame(X_validation)

# make sure y_validation is a factor with the same levels
y_validation <- factor(y_validation, levels = c(0, 1), labels = c("Class0", "Class1"))

# Prepare validation data
X_validation <- as.data.frame(X_validation)


# feature selection





# Training

# Set up cross-validation parameters
trainControl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = TRUE
)

metric <- "ROC"

# Train model with cross-validation
glm_model <- train( Class ~ .,data = X_train,method = "glm",family = "binomial",trControl = trainControl,metric = metric)

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
tuneGrid_knn <- expand.grid(k = seq(11, 17, by = 2))

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


# SVM


