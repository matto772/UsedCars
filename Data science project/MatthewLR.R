#Matthew Ogurkis
#MatthewLR.R - my attempt at linear regression for this problem
#Date started: 11/20/2024
#Date completed: 11/30/2024


#Load libraries
library(dplyr)
library(readr)
library(stringr)
library(caret)
library(ggplot2)

# Load the datasets
train <- read_csv('train.csv')
test_data <- read_csv('test.csv')


# Mappings for categorical features
fuel_map <- c('Gasoline' = 0, 'Hybrid' = 1, 'E85 Flex Fuel' = 2, 'missing' = 3, 
              'Diesel' = 4, '-' = 5, 'Plug-In Hybrid' = 6, 'not supported' = 7)

accident_map <- c('not_reported' = 0, 'reported' = 1, 'missing' = 2)

clean_map <- c('Yes' = 0, 'missing' = 1)

#Function to update the dataset
update <- function(df) {
  t <- 100
  
  #Updating 'accident' column with more consistent values
  df$accident <- recode(df$accident, 
                        `None reported` = 'not_reported', 
                        `At least 1 accident or damage reported` = 'reported')
  
  #Cleaning up the 'transmission' column
  df$transmission <- str_replace_all(df$transmission, "[-/]", "")
  
  #List of categorical columns to process
  cat_cols <- c('brand', 'model', 'fuel_type', 'engine', 'transmission', 
                'ext_col', 'int_col', 'accident', 'clean_title')
  rare_cols <- c('model', 'engine', 'transmission', 'ext_col', 'int_col')
  
  #Replace rare categories with a little bit of 'noise'
  for (col in rare_cols) {
    value_counts <- table(df[[col]], useNA = "ifany")
    rare_values <- names(value_counts[value_counts < t])
    df[[col]][df[[col]] %in% rare_values] <- 'noise'
  }
  
  #Fill in missing values and convert to factor
  for (col in cat_cols) {
    df[[col]] <- factor(ifelse(is.na(df[[col]]), 'missing', df[[col]]))
  }
  
  # Map categorical values to integers
  df$fuel_type <- as.integer(factor(df$fuel_type, levels = names(fuel_map), labels = fuel_map))
  df$accident <- as.integer(factor(df$accident, levels = names(accident_map), labels = accident_map))
  df$clean_title <- as.integer(factor(df$clean_title, levels = names(clean_map), labels = clean_map))
  
  return(df)
}

#Applying the update function to train and test datasets
train <- update(train)
test_data <- update(test_data)

#Remove outliers function
remove_outliers <- function(df, columns) {
  df_clean <- df
  
  for (col in columns) {
    #Check if the column exists in the data frame
    if (!(col %in% names(df_clean))) {
      cat(sprintf("Column '%s' does not exist in the data frame. Skipping...\n", col))
      next
    }
    
    #Check if the column is numeric
    if (!is.numeric(df_clean[[col]])) {
      cat(sprintf("Column '%s' is not numeric. Skipping...\n", col))
      next
    }
    
    #Calculate Q1, Q3, and IQR, while handling the NA values
    Q1 <- quantile(df_clean[[col]], 0.25, na.rm = TRUE)
    Q3 <- quantile(df_clean[[col]], 0.75, na.rm = TRUE)
    IQR <- Q3 - Q1
    lower_bound <- Q1 - 1.5 * IQR
    upper_bound <- Q3 + 1.5 * IQR
    
    #Filter the data frame to remove outliers
    df_clean <- df_clean %>% filter(df_clean[[col]] >= lower_bound & df_clean[[col]] <= upper_bound)
  }
  
  return(df_clean)
}

#Define numerical columns
numerical_col <- c('milage', 'model_year') 

#Preparing training data
train_clean <- remove_outliers(train, numerical_col)

#Train and predict using mil age
X_train_milage <- train_clean['milage']
y_train_milage <- train_clean$price

#Standardize the features for milage
scaler_milage <- preProcess(X_train_milage, method = c("center", "scale"))
X_train_milage_scaled <- predict(scaler_milage, X_train_milage)

#Combine the response variable with the scaled features for training
train_data_milage <- data.frame(price = y_train_milage, X_train_milage_scaled)

#Initialize and train the Linear Regression model using caret for milage
set.seed(123)
train_control_milage <- trainControl(method = "cv", number = 5)  
model_milage <- train(price ~ ., data = train_data_milage, method = "lm", trControl = train_control_milage)

#Predict on the training data to evaluate performance for milage
y_train_pred_milage <- predict(model_milage, newdata = train_data_milage)
rmse_train_milage <- sqrt(mean((y_train_milage - y_train_pred_milage)^2))
cat(sprintf('Linear Model RMSE on Training Data (Milage): %.4f\n', rmse_train_milage))

#Predict on the test data using milage
X_test_milage <- test_data['milage']
X_test_milage_scaled <- predict(scaler_milage, X_test_milage)
y_test_pred_milage <- predict(model_milage, newdata = as.data.frame(X_test_milage_scaled))

#Train and predict using model_year
X_train_model_year <- train_clean['model_year']
y_train_model_year <- train_clean$price

#Standardize the features for model_year
scaler_model_year <- preProcess(X_train_model_year, method = c("center", "scale"))
X_train_model_year_scaled <- predict(scaler_model_year, X_train_model_year)

#Combine the response variable with the scaled features for training
train_data_model_year <- data.frame(price = y_train_model_year, X_train_model_year_scaled)

#Initialize and train the Linear Regression model using caret for model_year
set.seed(123)
train_control_model_year <- trainControl(method = "cv", number = 5)  
model_model_year <- train(price ~ ., data = train_data_model_year, method = "lm", trControl = train_control_model_year)

#Predict on the training data to evaluate performance for model_year
y_train_pred_model_year <- predict(model_model_year, newdata = train_data_model_year)
rmse_train_model_year <- sqrt(mean((y_train_model_year - y_train_pred_model_year)^2))
cat(sprintf('Linear Model RMSE on Training Data (Model Year): %.4f\n', rmse_train_model_year))

#Predict on the test data using model_year
X_test_model_year <- test_data['model_year']
X_test_model_year_scaled <- predict(scaler_model_year, X_test_model_year)
y_test_pred_model_year <- predict(model_model_year, newdata = as.data.frame(X_test_model_year_scaled))

#Output predictions for both models
predictions <- data.frame(Milage_Predictions = y_test_pred_milage, Model_Year_Predictions = y_test_pred_model_year)

#Output predictions for both models
predictions <- data.frame(Milage_Predictions = y_test_pred_milage, Model_Year_Predictions = y_test_pred_model_year)

#Display the predictions
print(predictions)

#Scatter plot: Predicted vs Actual Prices (Mileage Model)
ggplot(data = data.frame(Actual = y_train_milage, Predicted = y_train_pred_milage), aes(x = Actual, y = Predicted)) +
  geom_point(color = "blue", alpha = 0.6) +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(
    title = "Predicted vs Actual Prices (Mileage Model)",
    x = "Actual Prices",
    y = "Predicted Prices"
  ) +
  theme_minimal()

#RMSE Comparison Bar Chart
rmse_data <- data.frame(
  Model = c("Mileage Model", "Model Year Model"),
  RMSE = c(rmse_train_milage, rmse_train_model_year)
)

ggplot(rmse_data, aes(x = Model, y = RMSE, fill = Model)) +
  geom_bar(stat = "identity", width = 0.6) +
  scale_fill_manual(values = c("Mileage Model" = "blue", "Model Year Model" = "green")) +
  labs(
    title = "RMSE Comparison for Mileage and Model Year Models",
    x = "Model",
    y = "RMSE"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

