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

# Separate majority and minority classes
majority_class <- pulsar[pulsar$X0 == 0, ]
minority_class <- pulsar[pulsar$X0 == 1, ]

# Random upsampling of minority class to match majority class
set.seed(123)
minority_upsampled <- minority_class[sample(nrow(minority_class), 
                                            size = nrow(majority_class), 
                                            replace = TRUE), ]

# Combine majority class with upsampled minority class
balanced_pulsar <- rbind(majority_class, minority_upsampled)

# Shuffle the balanced dataset
pulsar <- balanced_pulsar[sample(nrow(balanced_pulsar)), ]

# Check new class distribution
print("\nBalanced class distribution:")
table(pulsar$X0)
prop.table(table(pulsar$X0)) * 100



# Number of missing values

colSums(is.na(pulsar))


# Preprocessing

# Split the data into x and y
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

# Convert to factors with proper levels
y_train <- factor(y_train, levels = c(0, 1), labels = c("Class0", "Class1"))
y_validation <- factor(y_validation, levels = c(0, 1), labels = c("Class0", "Class1"))

# Convert back to dataframes
train_data <- as.data.frame(X_train)
X_validation <- as.data.frame(X_validation)

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

# Convert response to factor with proper level names
#y_train <- factor(y_train, levels = c(0, 1), labels = c("Class0", "Class1"))
#y_validation <- factor(y_validation, levels = c(0, 1), labels = c("Class0", "Class1"))

# Create training data frame
train_data <- data.frame(X_train)
train_data$Class <- y_train

# Train model with cross-validation
glm_model <- train(
  Class ~ .,
  data = train_data,
  method = "glm",
  family = "binomial",
  trControl = trainControl,
  metric = metric
)


summary (glm_model)

# Print results
print(glm_model)
print(glm_model$results)

# Make predictions on validation set
predictions <- predict(glm_model, newdata = as.data.frame(X_validation))
confusion_matrix <- confusionMatrix(predictions, y_validation)
print(confusion_matrix)


# glmnet
# Train model with cross-validation
glmnet_model <- train(
  Class ~ .,
  data = train_data,
  method = "glmnet",
  trControl = trainControl,
  metric = "ROC"
)

# Print results
print("Best tuning parameters:")
print(glmnet_model$bestTune)
print("\nModel performance across parameters:")
print(glmnet_model)

# Make predictions
predictions <- predict(glmnet_model, newdata = as.data.frame(X_validation))
confusion_matrix <- confusionMatrix(predictions, y_validation)
print("\nConfusion Matrix and Statistics:")
print(confusion_matrix)

# glmnet tuned
# Set up tuning grid for glmnet
tuneGrid <- expand.grid(
  alpha = seq(0, 1, by = 0.2),
  lambda = seq(0.0001, 0.1, length = 10)
)

# Train the tuned glmnet model
set.seed(123)  # for reproducibility
glmnet_tuned <- train(
  Class ~ .,
  data = train_data,
  method = "glmnet",
  trControl = trainControl,
  tuneGrid = tuneGrid,
  metric = "ROC"
)

# Make predictions using X_validation (not y_validation)
predictions <- predict(glmnet_tuned, newdata = X_validation)
prob_predictions <- predict(glmnet_tuned, newdata = X_validation, type = "prob")

# Create confusion matrix
confusion_matrix <- confusionMatrix(predictions, y_validation)
print("\nConfusion Matrix and Statistics:")
print(confusion_matrix)

