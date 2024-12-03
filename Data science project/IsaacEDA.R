# Isaac Waycott

par(mfrow = c(2, 2))
boxplot(test$model_year, main = "Model Year")
boxplot(test$milage, main = "Milage")
hist(test$model_year, main = "Model Year")
hist(test$milage, main = "Milage")


summary(test$milage)

summary(test$model_year)



milageOutliers = boxplot(test$milage, plot=FALSE)$out
modelYearOutliers = boxplot(test$model_year, plot=FALSE)$out


x = test
x = x[-which(x$model_year %in% modelYearOutliers),]
x = x[-which(x$milage %in% milageOutliers),]


summary(x$model_year)
 
summary(x$milage)



par(mfrow = c(2, 2))
boxplot(x$model_year, main = "Model Year")
boxplot(x$milage, main = "Milage")
hist(x$model_year, main = "Model Year")
hist(x$milage, main = "Milage")



# Code for KNN


install.packages(c("tidyverse", "caret", "class", "doParallel"))

library(tidyverse)

# Load the dataset
setwd("~/Downloads")
train_cleaned = read_csv("train_cleaned.csv")
test = read.csv("test.csv")
train = read.csv("train.csv")

# View the first few rows
head(train_cleaned)

# Check for missing values
colSums(is.na(train_cleaned))

# Convert categorical variables to factors (if applicable)
train_cleaned = train_cleaned %>%
  mutate_if(is.character, as.factor)

# Scale numeric variables
numeric_features = train_cleaned %>%
  select(where(is.numeric)) %>%
  scale()

# Combine scaled numeric features and categorical variables
train_cleaned = bind_cols(as.data.frame(numeric_features), train_cleaned %>% select(where(is.factor)))

# View processed data
str(train_cleaned)

library(caret)

set.seed(123) # For reproducibility
train_index = createDataPartition(train_cleaned$price, p = 0.8, list = FALSE)
train_set = train_cleaned[train_index, ]
test_set = train_cleaned[-train_index, ]
test_set = test_set %>%
  select(-id, -accident, -clean_title)

# Define the control method and enable progress updates
control <- trainControl(method = "cv", number = 5, verboseIter = TRUE)

library(doParallel)

# Register cores for parallel processing
cl <- makeCluster(4)  # Use 4 cores
registerDoParallel(cl)


# Train the KNN model
knn_model = train(price ~ ., 
                   data = train_set,
                   method = "knn",
                   trControl = control,
                   preProcess = c("center", "scale"),
                   tuneLength = 10)

# View model summary
print(knn_model)

# Use a smaller subset for testing
set.seed(123)
train_set_sample <- train_set %>% sample_n(500)  # Adjust sample size as needed
# Remove columns with levels less than or equal to 2
train_set_sample <- train_set_sample %>%
  select(-id, -accident, -clean_title)

# Train the model on the smaller dataset
knn_model <- train(price ~ ., 
                   data = train_set_sample,
                   method = "knn",
                   trControl = control,
                   preProcess = c("center", "scale"),
                   tuneLength = 10,
                   na.action = na.exclude)

# Stop parallel processing
stopCluster(cl)

# Make predictions
predictions = predict(knn_model, newdata = test_set)

# Evaluate performance
results = data.frame(Observed = test_set$price, Predicted = predictions)
print(cor(results)) # Correlation between observed and predicted

# Compute RMSE
rmse = sqrt(mean((results$Observed - results$Predicted)^2))
print(rmse)



# Code for KNN Cross Validation


# Install necessary packages (if not already installed)
install.packages(c("caret", "tidyverse", "doParallel"))

library(tidyverse)
library(caret)
library(doParallel)

# Load the dataset
setwd("~/Downloads")
train_data <- read_csv("train_cleaned.csv")

# Remove rows with missing values in specific columns
train_data <- train_data %>%
  filter(!is.na(fuel_type) & !is.na(accident) & !is.na(clean_title))

# Convert categorical variables to factors
train_data <- train_data %>%
  mutate_if(is.character, as.factor)

# Remove columns with levels less than or equal to 2
train_data <- train_data %>%
  select_if(function(col) !(is.factor(col) && length(unique(col)) <= 2))

# Check data structure
str(train_data)

set.seed(123)

# Define the target variable and predictors
target <- "price"
predictors <- setdiff(names(train_data), target)

# Create the training and validation sets
train_index <- createDataPartition(train_data[[target]], p = 0.8, list = FALSE)
train_set <- train_data[train_index, ]
test_set <- train_data[-train_index, ]

# Remove ID column
train_set <- train_set %>%
  select(-id)

# Define cross-validation settings
control <- trainControl(method = "cv", number = 5, verboseIter = TRUE) # 5-fold cross-validation

# Train the KNN model
set.seed(123)
knn_cv_model <- train(
  price ~ ., 
  data = train_set, 
  method = "knn",
  trControl = control,
  preProcess = c("center", "scale"), # Scale the data
  tuneLength = 10 # Test 10 values of k
)

# Print the model summary
print(knn_cv_model)

# Use a smaller subset for testing
set.seed(123)
train_set_sample <- train_set %>% sample_n(500)  # Adjust sample size as needed

# Train the model on the smaller dataset
knn_cv_model <- train(price ~ ., 
                   data = train_set_sample,
                   method = "knn",
                   trControl = control,
                   preProcess = c("center", "scale"),
                   tuneLength = 10,
                   na.action = na.exclude)

# Print the model summary
print(knn_cv_model)

# Predict on the test set
predictions <- predict(knn_cv_model, newdata = test_set)

# Calculate RMSE and R-squared
rmse <- sqrt(mean((predictions - test_set$price)^2))
r_squared <- cor(predictions, test_set$price)^2

cat("RMSE:", rmse, "\n")
cat("R-squared:", r_squared, "\n")



# Code for FNN(fast nearest neighbors)


# Install packages if not already installed
install.packages(c("FNN", "tidyverse"))

library(FNN)
library(tidyverse)

# Load the dataset
setwd("~/Downloads")
train_data <- read_csv("train_cleaned.csv")

# Remove rows with missing values in specified columns
train_data <- train_data %>%
  filter(!is.na(fuel_type) & !is.na(accident) & !is.na(clean_title))

# Remove columns with levels less than or equal to 2
train_data <- train_data %>%
  select_if(function(col) !(is.factor(col) && length(unique(col)) <= 2))

# Remove ID column
train_data <- train_data %>%
  select(-id)

# Convert categorical variables to factors and scale numeric data
train_data <- train_data %>%
  mutate_if(is.character, as.factor)

# Identify numeric columns and scale them
numeric_features <- train_data %>% 
  select(where(is.numeric)) %>% 
  scale()

# Combine scaled numeric features with non-numeric ones
train_data <- bind_cols(as.data.frame(numeric_features), 
                        train_data %>% select(where(is.factor)))

# Split data into predictors and target
target <- "price"
X <- train_data %>% select(-all_of(target))
y <- train_data[[target]]

set.seed(123)

# Create indices for the split
train_indices <- sample(1:nrow(X), size = 0.8 * nrow(X))
X_train <- X[train_indices, ]
y_train <- y[train_indices]

X_test <- X[-train_indices, ]
y_test <- y[-train_indices]

# Remove all non-numeric data from X_test and X_train
X_test <- X_test %>%
  select(-3, -4, -5, -6, -7, -8, -9, -10, -11)
X_train <- X_train %>%
  select(-3, -4, -5, -6, -7, -8, -9, -10, -11)

# Set the value of k
k_value <- 20  # Adjust this as needed

# Train the KNN regression model
knn_model <- knn.reg(train = X_train, test = X_test, y = y_train, k = k_value)

# Print the model summary
print(knn_model)

# Predictions
predictions <- knn_model$pred

# Calculate RMSE
rmse <- sqrt(mean((y_test - predictions)^2))

# Calculate R-squared
r_squared <- cor(y_test, predictions)^2

cat("RMSE:", rmse, "\n")
cat("R-squared:", r_squared, "\n")



# Find the best k using a range of values
best_k <- 1:20
errors <- sapply(best_k, function(k) {
  model <- knn.reg(train = X_train, test = X_test, y = y_train, k = k)
  sqrt(mean((model$pred - y_test)^2))  # RMSE for each k
})

# Plot RMSE vs. k
plot(best_k, errors, type = "b", xlab = "k", ylab = "RMSE", main = "Optimal k Selection")
