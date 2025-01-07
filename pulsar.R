# v8255920 James Fothergill
# Applied Machine Learning
# Pulsar Detection

library(caret)
library(e1071)
library(rpart)
library(randomForest)
library(nnet)
library(tidyr)
library(dplyr)
library(pROC)
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

set.seed(123)  # for reproducibility
glm_model <- caret::train( Class ~ .,data = X_train,method = "glm",family = "binomial",trControl = trainControl,metric = "ROC")

# Print results

summary (glm_model)
print(glm_model)
print(glm_model$results)

# Make predictions on validation set

predictions <- predict(glm_model, newdata = as.data.frame(X_validation))

# Create confusion matrix

confusion_matrixglm <- confusionMatrix(predictions, y_validation)
print("Confusion Matrix and Statistics glm: ")
print(confusion_matrixglm)


## glmnet

# Train model with cross-validation

set.seed(123)  # for reproducibility
glmnet_model <- caret::train(Class ~ .,data = X_train,method ="glmnet",trControl = trainControl,metric = "ROC")

# Print results

summary (glmnet_model)
plot(glmnet_model)

# Print results
print("Best tuning parameters:")
print(glmnet_model$bestTune)
print(glmnet_model)

# Make predictions

predictions <- predict(glmnet_model, newdata = as.data.frame(X_validation))

# Create confusion matrix

confusion_matrixglmn <- confusionMatrix(predictions, y_validation)
print("Confusion Matrix and Statistics glmnet:")
print(confusion_matrixglmn)



## glmnet ridge tuned

# Set up tuning grid for glmnet ridge

tuneGrid_r <- expand.grid(alpha = 0, lambda = seq(0.01, 0.001, length = 10)) # Best settings


# Train the tuned glmnet model ridge

set.seed(123)  # for reproducibility
glmnet_tuned_r <- caret::train(Class ~ .,data = X_train,method = "glmnet",trControl = trainControl,tuneGrid = tuneGrid_r,metric = "ROC")

# Print results

summary (glmnet_tuned_r)
plot(glmnet_tuned_r)
print(glmnet_tuned_r$bestTune)

# Make predictions

predictions <- predict(glmnet_tuned_r, newdata = X_validation)

# Create confusion matrix

confusion_matrixglmr <- confusionMatrix(predictions, y_validation)
print("Confusion Matrix and Statistics ridge:")
print(confusion_matrixglmr)



## glmnet lasso tuned

# Set up tuning grid for glmnet lasso

tuneGrid_l <- expand.grid(alpha = 1 ,lambda = 0.01) # Best settings

# Train the tuned glmnet model lasso

set.seed(123)  # for reproducibility
glmnet_tuned_l <- caret::train(Class ~ .,data = X_train,method = "glmnet",trControl = trainControl,tuneGrid = tuneGrid_l,metric = "ROC")

# Print results

summary (glmnet_tuned_l)

# Make predictions

predictions <- predict(glmnet_tuned_l, newdata = X_validation)

# Create confusion matrix

confusion_matrixglml <- confusionMatrix(predictions, y_validation)
print("Confusion Matrix and Statistics lasso:")
print(confusion_matrixglml)

# glmnet elastic net

# Set up tuning grid for glmnet elastic

tuneGrid_e <- expand.grid(alpha = seq(0.5, 0.8, by = 0.3),lambda = seq(0.1, 0.001, length = 10)) # Best settings

# Train the tuned glmnet model elastic net

set.seed(123)  # for reproducibility
glmnet_tuned_e <- caret::train(Class ~ .,data = X_train,method = "glmnet",trControl = trainControl,tuneGrid = tuneGrid_e,metric = "ROC")

# Print results

summary (glmnet_tuned_e)
plot(glmnet_tuned_e)
print(glmnet_tuned_e$bestTune)

# Make predictions

predictions <- predict(glmnet_tuned_e, newdata = X_validation)

# Create confusion matrix

confusion_matrixglme <- confusionMatrix(predictions, y_validation)
print("Confusion Matrix and Statistics Elastic net:")
print(confusion_matrixglme)



# K-NN

# KNN-specific cross-validation

trainControlknn <- trainControl(method = "repeatedcv",number = 10,repeats = 3,classProbs = TRUE,summaryFunction = twoClassSummary)

# KNN

set.seed(123)  # for reproducibility
knn_model <- caret::train(Class ~ .,data = X_train,method = "knn",trControl = trainControlknn,metric = "ROC")

# Print results

summary (knn_model)

# Make predictions

predictions <- predict(knn_model, newdata = X_validation)

# Create confusion matrix

confusion_matrixknn <- confusionMatrix(predictions, y_validation)
print("Confusion Matrix and Statistics K-NN:")
print(confusion_matrixknn)

# K-NN tuned


# Create tuning grid for KNN

tuneGrid_knn <- expand.grid(k = seq(1, 30, by = 1))

set.seed(123)  # for reproducibility
knn_model_t <- caret::train(Class ~ .,data = X_train,method = "knn",trControl = trainControlknn,tuneGrid = tuneGrid_knn,metric = "ROC") # was worse with PCA

summary (knn_model_t) 
plot(knn_model_t)
print(knn_model_t$bestTune)

# Make predictions

predictions <- predict(knn_model_t, newdata = X_validation)

# Create confusion matrix

confusion_matrixknnt <- confusionMatrix(predictions, y_validation)
print("Confusion Matrix and Statistics K-NN tuned:")
print(confusion_matrixknnt)


## K-NN best tuned = 13

# Create tuning grid for KNN

tuneGrid_knn <- expand.grid(k = 13)

set.seed(123)  # for reproducibility
knn_model_13 <- caret::train(Class ~ .,data = X_train,method = "knn",trControl = trainControlknn, tuneGrid = tuneGrid_knn,metric = "ROC") # was worse with PCA

summary (knn_model_13) 

# Make predictions

predictions <- predict(knn_model_13, newdata = X_validation)

# Create confusion matrix

confusion_matrixknn13 <- confusionMatrix(predictions, y_validation)
print("Confusion Matrix and Statistics K-NN tuned 13:")
print(confusion_matrixknn13)





## SVM radial Tuned

tuneGrid_svm_r <- expand.grid(sigma = seq(0.01, 2.5, length = 8), C = seq(0.01, 8, length = 8))

# Tune the model

set.seed(123) # for reproducibility
svm_radial <- caret::train(Class ~ .,data = X_train,method = "svmRadial",trControl = trainControl(method = "repeatedcv",number = 10,repeats = 3,classProbs = TRUE,summaryFunction = twoClassSummary),
                    tuneGrid = tuneGrid_svm_r, metric = "ROC")

# Print results

print(svm_radial)
plot(svm_radial) 
print(svm_radial$bestTune) 

# Make predictions

predictions <- predict(svm_radial, newdata = X_validation)

# Create confusion matrix

confusion_matrix_svm <- confusionMatrix(predictions, y_validation)
print("Confusion Matrix and Statistics SVM Radial:")
print(confusion_matrix_svm)




## SVM Linear Tuned

tuneGrid_svm_l <- expand.grid(C = seq(0.01, 8, length = 8))

# Tune the model

set.seed(123)
svm_linear <- caret::train(Class ~ .,data = X_train,method = "svmLinear",trControl = trainControl(method = "repeatedcv",number = 10,repeats = 3,classProbs = TRUE,summaryFunction = twoClassSummary),
                    tuneGrid = tuneGrid_svm_l,metric = "ROC")

# Print results

print(svm_linear)
plot(svm_linear)
print(svm_linear$bestTune)

# Make predictions

predictions <- predict(svm_linear, newdata = X_validation)

# Create confusion matrix

confusion_matrix_svml <- confusionMatrix(predictions, y_validation)
print("Confusion Matrix and Statistics SVM Linear:")
print(confusion_matrix_svml)






## RANDOM FOREST

# Set up cross-validation parameters


trainControl_rf <- trainControl(method = "repeatedcv", number = 10,repeats = 3, classProbs = TRUE, summaryFunction = twoClassSummary)


# Train Random Forest

set.seed(123)  # for reproducibility
rf_model <- caret::train(Class ~ .,data = X_train,method = "rf",trControl = trainControl_rf,metric = "ROC")

# Print results

print(rf_model)
plot(rf_model)

# Make predictions

rf_predictions <- predict(rf_model, newdata = X_validation)

# Create confusion matrix for the Random Forest model

confusion_matrixrf <- confusionMatrix(rf_predictions, y_validation)
print("Confusion Matrix and Statistics for Random Forest:")
print(confusion_matrixrf)




## Random Forest Tuning grid


# tuning grid for mtry

tuneGrid_rf <- expand.grid(mtry = seq(1, 8, by = 1))

# Train the Random Forest model with tuning

set.seed(123)  # for reproducibility
rf_model_tuned <- caret::train(Class ~ .,data = X_train,method = "rf",trControl = trainControl_rf,tuneGrid = tuneGrid_rf,
                               metric = "ROC",ntree = 80,nodesize = 2,maxnodes = 1000)

# Print results

print(rf_model_tuned)
best_mtry <- rf_model_tuned$bestTune$mtry 
print(paste("Best mtry value:", best_mtry))
plot(rf_model_tuned)

# Make predictions

rf_predictions_tuned <- predict(rf_model_tuned, newdata = X_validation)

# Create confusion matrix for the tuned Random Forest model

confusion_matrix_tunedrf <- confusionMatrix(rf_predictions_tuned, y_validation)
print("Confusion Matrix and Statistics for Tuned Random Forest tuned:")
print(confusion_matrix_tunedrf)


## Final model Random Forest Tuned


tuneGrid_rf <- expand.grid(mtry = 1)

# Train the Random Forest model with tuning

set.seed(123)  # for reproducibility
rf_model_final <- caret::train(Class ~ .,data = X_train,method = "rf",trControl = trainControl_rf,tuneGrid = tuneGrid_rf,
                               metric = "ROC",ntree = 80,nodesize = 2,maxnodes = 1000)


# Print results

print(rf_model_final)

# Make predictions

rf_predictions_final <- predict(rf_model_final, newdata = X_validation)

# Create confusion matrix for the tuned Random Forest model

confusion_matrix_tunedrffinal <- confusionMatrix(rf_predictions_final, y_validation)
print("Confusion Matrix and Statistics for Tuned Random Forest final:")
print(confusion_matrix_tunedrffinal)




## ROC curve for best models

# Get probabilities

# GLM probabilities
glm_pred_prob <- predict(glmnet_tuned_l, newdata = X_validation, type = "prob")[,"Class1"]

# Linear SVM probabilities
svm_linear_pred_prob <- predict(svm_linear, newdata = X_validation, type = "prob")[,"Class1"]

# RF Final probabilities
rf_pred_prob <- predict(rf_model_final, newdata = X_validation, type = "prob")[,"Class1"]

# KNN  probabilities
knn_pred_prob <- predict(knn_model, newdata = X_validation, type = "prob")[,"Class1"]

# Create ROC objects
library(pROC)
roc_glm <- roc(y_validation, glm_pred_prob)
roc_svm_linear <- roc(y_validation, svm_linear_pred_prob)
roc_rf <- roc(y_validation, rf_pred_prob)
roc_knn <- roc(y_validation, knn_pred_prob)

# Plot ROC curves
plot.roc(roc_rf, 
         col = "red",
         main = "ROC Curves Comparison",
         lwd = 2)

plot.roc(roc_glm, 
         col = "blue", 
         add = TRUE, 
         lwd = 2)

plot.roc(roc_svm_linear, 
         col = "green", 
         add = TRUE, 
         lwd = 2)

plot.roc(roc_knn, 
         col = "purple", 
         add = TRUE, 
         lwd = 2)

# Add legend
legend("bottomright", 
       legend = c(paste0("Random Forest (AUC = ", round(auc(roc_rf), 3), ")"),
                  paste0("GLMnet Lasso (AUC = ", round(auc(roc_glm), 3), ")"),
                  paste0("SVM Linear (AUC = ", round(auc(roc_svm_linear), 3), ")"),
                  paste0("KNN Base (AUC = ", round(auc(roc_knn), 3), ")")),
       col = c("red", "blue", "green", "purple"),
       lwd = 2)



## Plot for top accuracy models

# accuracy from confusion matrices

glm_accuracy <- confusion_matrixglml$overall['Accuracy']
rf_accuracy <- confusion_matrix_tunedrf$overall['Accuracy']
knn_accuracy <- confusion_matrixknn$overall['Accuracy']
svm_accuracy <- confusion_matrix_svml$overall['Accuracy']

# data frame with values

model_performance <- data.frame(
  Model = c("GLMnet Lasso", "Random Forest Final", "KNN Base", "SVM Linear"),
  Accuracy = c(glm_accuracy, rf_accuracy, knn_accuracy, svm_accuracy)
)

# Accuracy plot

ggplot(model_performance, aes(x = reorder(Model, -Accuracy), y = Accuracy)) +
  geom_bar(stat = "identity", 
           fill = c("#619CFF", "#F8766D", "#00BA38", "#9590FF"),
           width = 0.6) +
  geom_text(aes(label = sprintf("%.3f", Accuracy)), 
            vjust = -0.5) +
  ylim(0, 1) +
  theme_minimal() +
  labs(title = "Model Accuracy Comparison",
       y = "Accuracy Score",
       x = NULL) +
  theme(plot.title = element_text(hjust = 0.5),
        panel.grid.major.x = element_blank())