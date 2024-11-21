#Jackie Campbell
# Load necessary libraries
library(tidyverse)
library(glmnet)  # For Lasso regression

# Set plotting layout
par(mfrow = c(2, 2))

# Plot initial boxplots and histograms for model year and mileage
boxplot(test$model_year, main = "Model Year", ylab = "Year")
boxplot(test$milage, main = "Mileage", ylab = "Milage (miles)")
hist(test$model_year, main = "Model Year", xlab = "Year", col = "lightblue", breaks = 20)
hist(test$milage, main = "Mileage", xlab = "Milage (miles)", col = "lightgreen", breaks = 20)

# Display summary statistics for mileage and model year
summary(test$milage)
summary(test$model_year)

# Identify outlines using the box plot statistics
milageOutliers <- boxplot(test$milage, plot = FALSE)$out
modelYearOutliers <- boxplot(test$model_year, plot = FALSE)$out

# Create a copy of the test dataset for cleaning
x <- test

# Remove rows with outliers in model year
x <- x[!x$model_year %in% modelYearOutliers, ]

# Remove rows with outliers in mileage
x <- x[!x$milage %in% milageOutliers, ]

# Display summary statistics after removing outliers
summary(x$model_year)
summary(x$milage)

# Plot boxplots and histograms after outlier removal
par(mfrow = c(2, 2))
boxplot(x$model_year, main = "Model Year (Cleaned)", ylab = "Year")
boxplot(x$milage, main = "Mileage (Cleaned)", ylab = "Mileage (miles)")
hist(x$model_year, main = "Model Year (Cleaned)", xlab = "Year", col = "lightblue", breaks = 20)
hist(x$milage, main = "Mileage (Cleaned)", xlab = "Mileage (miles)", col = "lightgreen", breaks = 20)

# Assuming 'x' is the cleaned dataset from your code
# Add a dummy target variable 'car_cost' for illustration (replace with actual data)
set.seed(123)  # For reproducibility
x$car_cost <- 20000 - 100 * x$model_year + 0.5 * x$milage + rnorm(nrow(x), mean = 0, sd = 500)

# Prepare data for Lasso regression
# Extract predictors and target variable
X <- model.matrix(car_cost ~ model_year + milage, data = x)[, -1]  # Remove intercept
y <- x$car_cost

# Split data into training and testing sets
set.seed(123)
train_index <- sample(1:nrow(x), size = 0.8 * nrow(x))
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

# Fit Lasso regression model
lasso_model <- cv.glmnet(X_train, y_train, alpha = 1, nfolds = 10)  # Lasso (alpha = 1)

# Best lambda value from cross-validation
best_lambda <- lasso_model$lambda.min
cat("Best Lambda:", best_lambda, "\n")

# Plot cross-validation results
plot(lasso_model)

# Make predictions on the test set
y_pred <- predict(lasso_model, s = best_lambda, newx = X_test)

# Evaluate model performance
mse <- mean((y_test - y_pred)^2)
cat("Mean Squared Error (MSE) on Test Data:", mse, "\n")

# Check coefficients of the model
lasso_coefficients <- coef(lasso_model, s = best_lambda)
print("Coefficients:")
print(lasso_coefficients)
