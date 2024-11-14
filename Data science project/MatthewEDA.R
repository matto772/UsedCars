# EDA.R
# Matthew Ogurkis
# Date Started: 11/9/2024
# Date Completed: 11/13/2024

# Loading libraries
library(dplyr)
library(ggplot2)
library(corrplot)
library(tidyr)
library(readr)
library(stringr)

# Loading the dataset
data <- read_csv("train.csv")

# Data Cleaning
# Converting character columns to factors
data <- data %>%
  mutate(across(where(is.character), as.factor))

#Replace NAs in numeric columns with mean and in categorical with the mode
replace_na_number <- function(x) {
  if (is.numeric(x)) x[is.na(x)] <- mean(x, na.rm = TRUE)
  return(x)
}
replace_na_categorical <- function(x) {
  if (is.factor(x)) {
    mode_value <- names(sort(table(x), decreasing = TRUE))[1]
    x[is.na(x)] <- mode_value
  }
  return(x)
}
data <- data %>%
  mutate(across(where(is.numeric), replace_na_number)) %>%
  mutate(across(where(is.factor), replace_na_categorical))

data <- data %>%
  filter(between(price, quantile(price, 0.05), quantile(price, 0.95)),
         between(milage, quantile(milage, 0.05), quantile(milage, 0.95))) %>%
  mutate(price = ifelse(abs(scale(price)) > 3, mean(price, na.rm = TRUE), price),
         milage = ifelse(abs(scale(milage)) > 3, mean(milage, na.rm = TRUE), milage))


# Box Plots
ggplot(data, aes(x = reorder(brand, price, median), y = price)) +
  geom_boxplot(fill = "lightblue") + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(title = "Boxplot of Car Prices by Brand (Filtered)", x = "Brand", y = "Price")

ggplot(data, aes(x = fuel_type, y = price)) +
  geom_boxplot(fill = "lightgreen") +
  labs(title = "Boxplot of Car Prices by Fuel Type (Filtered)", x = "Fuel Type", y = "Price")

ggplot(data, aes(x = transmission, y = price)) +
  geom_boxplot(fill = "lightpink") +
  labs(title = "Boxplot of Car Prices by Transmission (Filtered)", x = "Transmission", y = "Price")

#histograms
#Histogram for 'price'
ggplot(data, aes(x = price)) + 
  geom_histogram(bins = 50, fill = "skyblue", color = "black") +
  labs(title = "Distribution of Car Prices", x = "Count", y = "Price")

#Histogram for 'milage'
ggplot(data, aes(x = milage)) + 
  geom_histogram(bins = 50, fill = "lightgreen", color = "black") +
  labs(title = "Distribution of Mileage", x = "Mileage", y = "Count")

#Histogram for 'model_year'
ggplot(data, aes(x = model_year)) + 
  geom_histogram(bins = 30, fill = "lightcoral", color = "black") +
  labs(title = "Distribution of Model Year", x = "Model Year", y = "Count")

#Count plot for 'brand'
ggplot(data, aes(x = reorder(brand, brand, function(x) -length(x)))) +
  geom_bar(fill = "steelblue") + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(title = "Count of Cars by Brand", x = "Brand", y = "Count")

#Count plot of 'fuel_type'
ggplot(data, aes(x = fuel_type)) + 
  geom_bar(fill = "purple") +
  labs(title = "Count of Cars by Fuel Type", x = "Fuel Type", y = "Count")

#Count plot of 'accident' status
ggplot(data, aes(x = accident)) + 
  geom_bar(fill = "orange") +
  labs(title = "Accident Reported Status", x = "Accident", y = "Count")

#Count plot of 'clean_title'
ggplot(data, aes(x = clean_title)) + 
  geom_bar(fill = "brown") +
  labs(title = "Clean Title Status", x = "Clean Title", y = "Count")

#writing the cleaned up data to a new csv
write.csv('cleaned_data.csv')

